"""Copy bundled agent skills into the directories agent clients scan.

Standard library only, and free of any ``tvboptim`` runtime import: this module
runs as the first thing a user does after installing the wheel, and it must not
depend on the numerical stack it ships beside.

Agent clients agree on the skill bundle format but not on where they look for
it. Only the destination differs; the copied bundle is byte-identical for every
agent. ``AGENT_DIRECTORIES`` is therefore the only place that knows about
vendors, and ``destination=`` bypasses it entirely so a new client never
requires a TVB-Optim release.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parent

#: Written inside an installed bundle so status, update, and uninstall can tell
#: an unmodified managed copy from one the user has edited by hand.
MANIFEST_NAME = ".tvboptim-skill.json"

#: Relative directory each client scans, per scope. Project paths are relative
#: to the project root; user paths are relative to the home directory.
AGENT_DIRECTORIES: dict[str, dict[str, str]] = {
    "agents": {"project": ".agents/skills", "user": ".agents/skills"},
    "codex": {"project": ".agents/skills", "user": ".agents/skills"},
    "cursor": {"project": ".agents/skills", "user": ".agents/skills"},
    "copilot": {"project": ".agents/skills", "user": ".agents/skills"},
    "claude-code": {"project": ".claude/skills", "user": ".claude/skills"},
}

AGENTS = tuple(AGENT_DIRECTORIES)
SCOPES = ("project", "user")


class SkillError(Exception):
    """Raised when a skill cannot be resolved, installed, or removed."""


@dataclass(frozen=True)
class Manifest:
    """Provenance recorded inside an installed bundle."""

    skill: str
    package_version: str
    content_hash: str
    installed_at: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "skill": self.skill,
                "package_version": self.package_version,
                "content_hash": self.content_hash,
                "installed_at": self.installed_at,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_path(cls, path: Path) -> Manifest | None:
        """Read a manifest, returning ``None`` if absent or unreadable."""
        try:
            raw = json.loads(path.read_text())
            return cls(
                skill=raw["skill"],
                package_version=raw["package_version"],
                content_hash=raw["content_hash"],
                installed_at=raw["installed_at"],
            )
        except (OSError, ValueError, KeyError):
            return None


@dataclass(frozen=True)
class Installation:
    """State of one destination directory."""

    path: Path
    exists: bool
    manifest: Manifest | None = None
    content_hash: str | None = None

    @property
    def managed(self) -> bool:
        """True when this copy was written by us and carries a manifest."""
        return self.manifest is not None

    @property
    def modified(self) -> bool:
        """True when a managed copy no longer matches its recorded hash."""
        if self.manifest is None or self.content_hash is None:
            return False
        return self.manifest.content_hash != self.content_hash

    @property
    def state(self) -> str:
        if not self.exists:
            return "absent"
        if not self.managed:
            return "unmanaged"
        if self.modified:
            return "modified"
        return "installed"


@dataclass(frozen=True)
class Result:
    """Outcome of an install, export, or uninstall."""

    action: str
    skill: str
    destination: Path
    dry_run: bool = False
    previous: Installation | None = None
    notes: list[str] = field(default_factory=list)


def package_version() -> str:
    """Version of the installed distribution, or ``"unknown"`` in a checkout."""
    try:
        from importlib.metadata import version

        return version("tvboptim")
    except Exception:
        return "unknown"


def available_skills() -> tuple[str, ...]:
    """Names of the skill bundles shipped inside this package."""
    return tuple(
        sorted(
            path.parent.name
            for path in SKILLS_ROOT.glob("*/SKILL.md")
            if path.is_file()
        )
    )


def skill_source(skill: str = "tvboptim") -> Path:
    """Directory holding a bundled skill.

    Raises
    ------
    SkillError
        If the named skill is not bundled, which is also how a wheel that
        dropped the package data announces itself.
    """
    source = SKILLS_ROOT / skill
    if not (source / "SKILL.md").is_file():
        known = ", ".join(available_skills()) or "none"
        raise SkillError(
            f"no bundled skill named {skill!r} (available: {known}). "
            "If tvboptim was installed from a wheel, its package data may be "
            "incomplete; please report this."
        )
    return source


def bundle_hash(directory: Path) -> str:
    """Content hash over a bundle, ignoring the manifest itself.

    Covers relative paths as well as bytes so that renaming or removing a
    reference changes the hash.
    """
    digest = hashlib.sha256()
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    for path in files:
        if path.name == MANIFEST_NAME and path.parent == directory:
            continue
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def inspect(path: Path) -> Installation:
    """Describe whatever currently occupies a destination."""
    if not path.exists():
        return Installation(path=path, exists=False)
    manifest = Manifest.from_path(path / MANIFEST_NAME)
    return Installation(
        path=path,
        exists=True,
        manifest=manifest,
        content_hash=bundle_hash(path) if manifest is not None else None,
    )


def resolve_destination(
    skill: str = "tvboptim",
    *,
    agent: str | None = None,
    scope: str = "project",
    project: Path | None = None,
    destination: Path | None = None,
) -> Path:
    """Map an agent and scope onto the directory to write.

    An explicit ``destination`` wins and is used verbatim, so an unsupported or
    newly released client never has to wait for a mapping update.
    """
    if destination is not None:
        return Path(destination).expanduser().resolve()

    if agent is None:
        raise SkillError(
            "specify --agent (one of: " + ", ".join(AGENTS) + ") or --destination"
        )
    if agent not in AGENT_DIRECTORIES:
        raise SkillError(
            f"unknown agent {agent!r}; expected one of: {', '.join(AGENTS)}. "
            "Use --destination to install somewhere not covered by this mapping."
        )
    if scope not in SCOPES:
        raise SkillError(
            f"unknown scope {scope!r}; expected one of: {', '.join(SCOPES)}"
        )

    relative = AGENT_DIRECTORIES[agent][scope]
    if scope == "user":
        root = Path.home()
    else:
        root = Path(project).expanduser() if project is not None else Path.cwd()
    return (root / relative / skill).resolve()


def _copy_atomically(source: Path, target: Path, manifest: Manifest) -> None:
    """Stage a copy beside the target, then swap it in with a rename.

    The bundle is copied rather than symlinked because a symlink into
    ``site-packages`` breaks the moment the wheel is upgraded or removed.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.tmp-"))
    try:
        staged = staging / target.name
        shutil.copytree(source, staged)
        (staged / MANIFEST_NAME).write_text(manifest.to_json() + "\n")

        if target.exists():
            os.rename(target, staging / "__replaced__")
        os.rename(staged, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install(
    agent: str | None = None,
    scope: str = "project",
    *,
    skill: str = "tvboptim",
    project: Path | None = None,
    destination: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> Result:
    """Copy a bundled skill into the directory an agent client scans.

    Refuses to overwrite a directory we did not write, or a managed copy that
    has been edited locally, unless ``force`` is set.
    """
    source = skill_source(skill)
    target = resolve_destination(
        skill, agent=agent, scope=scope, project=project, destination=destination
    )
    previous = inspect(target)
    notes: list[str] = []

    if previous.exists and not force:
        if not previous.managed:
            raise SkillError(
                f"{target} already exists and was not installed by tvboptim. "
                "Move it aside or pass --force to replace it."
            )
        if previous.modified:
            raise SkillError(
                f"{target} has local modifications since it was installed. "
                "Pass --force to discard them."
            )

    if previous.exists:
        if previous.managed and not previous.modified:
            notes.append(
                f"replacing {previous.manifest.package_version} installation"
                if previous.manifest
                else "replacing existing installation"
            )
        else:
            notes.append("overwriting (forced)")

    if not dry_run:
        manifest = Manifest(
            skill=skill,
            package_version=package_version(),
            content_hash=bundle_hash(source),
            installed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _copy_atomically(source, target, manifest)

    return Result(
        action="install",
        skill=skill,
        destination=target,
        dry_run=dry_run,
        previous=previous,
        notes=notes,
    )


def export(
    destination: Path,
    *,
    skill: str = "tvboptim",
    dry_run: bool = False,
    force: bool = False,
) -> Result:
    """Copy a bundled skill into ``destination/<skill>`` with no agent mapping.

    The permanent fallback for clients this package has never heard of.
    """
    target = (Path(destination).expanduser() / skill).resolve()
    previous = inspect(target)

    if previous.exists and not force:
        if not previous.managed:
            raise SkillError(
                f"{target} already exists and was not written by tvboptim. "
                "Choose another destination or pass --force."
            )
        if previous.modified:
            raise SkillError(
                f"{target} has local modifications. Pass --force to discard them."
            )

    if not dry_run:
        source = skill_source(skill)
        manifest = Manifest(
            skill=skill,
            package_version=package_version(),
            content_hash=bundle_hash(source),
            installed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _copy_atomically(source, target, manifest)

    return Result(
        action="export",
        skill=skill,
        destination=target,
        dry_run=dry_run,
        previous=previous,
    )


def uninstall(
    agent: str | None = None,
    scope: str = "project",
    *,
    skill: str = "tvboptim",
    project: Path | None = None,
    destination: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> Result:
    """Remove an installed skill, preserving anything we did not write."""
    target = resolve_destination(
        skill, agent=agent, scope=scope, project=project, destination=destination
    )
    previous = inspect(target)

    if not previous.exists:
        return Result(
            action="uninstall",
            skill=skill,
            destination=target,
            dry_run=dry_run,
            previous=previous,
            notes=["nothing to remove"],
        )
    if not force:
        if not previous.managed:
            raise SkillError(
                f"{target} was not installed by tvboptim; refusing to remove it. "
                "Pass --force to delete it anyway."
            )
        if previous.modified:
            raise SkillError(
                f"{target} has local modifications; refusing to remove it. "
                "Pass --force to delete it anyway."
            )

    if not dry_run:
        shutil.rmtree(target)

    return Result(
        action="uninstall",
        skill=skill,
        destination=target,
        dry_run=dry_run,
        previous=previous,
    )


def status(
    *,
    skill: str = "tvboptim",
    project: Path | None = None,
) -> list[Installation]:
    """Report every known destination that currently holds this skill.

    Deduplicated by path, because several agents share ``.agents/skills``.
    """
    seen: dict[Path, Installation] = {}
    for agent in AGENTS:
        for scope in SCOPES:
            try:
                path = resolve_destination(
                    skill, agent=agent, scope=scope, project=project
                )
            except (SkillError, RuntimeError):
                continue
            if path not in seen:
                seen[path] = inspect(path)
    return [seen[path] for path in sorted(seen)]

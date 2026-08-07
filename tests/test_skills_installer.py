"""Tests for the bundled agent skill installer and its CLI."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tvboptim import cli
from tvboptim.skills import _installer
from tvboptim.skills._installer import (
    AGENT_DIRECTORIES,
    AGENTS,
    SCOPES,
    SkillError,
    available_skills,
    bundle_hash,
    export,
    inspect,
    install,
    resolve_destination,
    skill_source,
    status,
    uninstall,
)

ROOT = Path(__file__).parents[1]


def test_bundled_skill_is_discoverable_and_complete():
    assert "tvboptim" in available_skills()
    source = skill_source("tvboptim")
    assert (source / "SKILL.md").is_file()
    assert sorted(path.name for path in (source / "references").glob("*.md"))


def test_unknown_skill_names_are_rejected():
    with pytest.raises(SkillError, match="no bundled skill"):
        skill_source("does-not-exist")


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("scope", SCOPES)
def test_every_agent_and_scope_resolves(agent, scope, tmp_path):
    target = resolve_destination("tvboptim", agent=agent, scope=scope, project=tmp_path)
    assert target.name == "tvboptim"
    assert target.parent.name == "skills"
    if scope == "project":
        assert target.is_relative_to(tmp_path)
    else:
        assert target.is_relative_to(Path.home())


@pytest.mark.parametrize("agent", AGENTS)
def test_project_mapping_matches_declared_directory(agent, tmp_path):
    relative = AGENT_DIRECTORIES[agent]["project"]
    target = resolve_destination(
        "tvboptim", agent=agent, scope="project", project=tmp_path
    )
    assert target == (tmp_path / relative / "tvboptim").resolve()


def test_destination_overrides_the_agent_mapping(tmp_path):
    explicit = tmp_path / "somewhere" / "else"
    assert resolve_destination("tvboptim", destination=explicit) == explicit.resolve()


def test_resolution_requires_an_agent_or_destination():
    with pytest.raises(SkillError, match="--agent"):
        resolve_destination("tvboptim")


def test_unknown_agent_is_rejected_with_guidance():
    with pytest.raises(SkillError, match="Use --destination"):
        resolve_destination("tvboptim", agent="emacs")


@pytest.mark.parametrize("agent", AGENTS)
def test_install_copies_the_whole_bundle(agent, tmp_path):
    result = install(agent=agent, scope="project", project=tmp_path)
    source = skill_source("tvboptim")

    assert result.destination.is_dir()
    assert (result.destination / "SKILL.md").read_text() == (
        source / "SKILL.md"
    ).read_text()

    expected = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    installed = {
        path.relative_to(result.destination).as_posix()
        for path in result.destination.rglob("*")
        if path.is_file() and path.name != _installer.MANIFEST_NAME
    }
    assert installed == expected
    assert bundle_hash(result.destination) == bundle_hash(source)


def test_install_is_a_copy_not_a_symlink(tmp_path):
    # A symlink into site-packages breaks when the wheel is upgraded or removed.
    result = install(agent="codex", scope="project", project=tmp_path)
    assert not result.destination.is_symlink()
    assert not (result.destination / "SKILL.md").is_symlink()


def test_install_records_provenance(tmp_path):
    result = install(agent="codex", scope="project", project=tmp_path)
    state = inspect(result.destination)

    assert state.managed
    assert not state.modified
    assert state.state == "installed"
    assert state.manifest.skill == "tvboptim"
    assert state.manifest.content_hash == bundle_hash(skill_source("tvboptim"))


def test_dry_run_writes_nothing(tmp_path):
    result = install(agent="codex", scope="project", project=tmp_path, dry_run=True)
    assert result.dry_run
    assert not result.destination.exists()


def test_reinstall_over_an_unmodified_copy_succeeds(tmp_path):
    first = install(agent="codex", scope="project", project=tmp_path)
    second = install(agent="codex", scope="project", project=tmp_path)
    assert second.destination == first.destination
    assert inspect(second.destination).state == "installed"


def test_install_refuses_to_clobber_an_unmanaged_directory(tmp_path):
    target = resolve_destination(
        "tvboptim", agent="codex", scope="project", project=tmp_path
    )
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("hand written")

    with pytest.raises(SkillError, match="not installed by tvboptim"):
        install(agent="codex", scope="project", project=tmp_path)
    assert (target / "SKILL.md").read_text() == "hand written"

    install(agent="codex", scope="project", project=tmp_path, force=True)
    assert (target / "SKILL.md").read_text() != "hand written"


def test_install_refuses_to_discard_local_modifications(tmp_path):
    result = install(agent="codex", scope="project", project=tmp_path)
    (result.destination / "SKILL.md").write_text("locally edited")
    assert inspect(result.destination).state == "modified"

    with pytest.raises(SkillError, match="local modifications"):
        install(agent="codex", scope="project", project=tmp_path)
    assert (result.destination / "SKILL.md").read_text() == "locally edited"

    install(agent="codex", scope="project", project=tmp_path, force=True)
    assert (result.destination / "SKILL.md").read_text() != "locally edited"


def test_failed_install_leaves_no_staging_directory(tmp_path):
    install(agent="codex", scope="project", project=tmp_path)
    target = resolve_destination(
        "tvboptim", agent="codex", scope="project", project=tmp_path
    )
    (target / "SKILL.md").write_text("locally edited")
    with pytest.raises(SkillError):
        install(agent="codex", scope="project", project=tmp_path)

    leftovers = [path for path in target.parent.iterdir() if path.name.startswith(".")]
    assert leftovers == []


def test_export_needs_no_agent(tmp_path):
    result = export(tmp_path / "bundle")
    assert result.destination == (tmp_path / "bundle" / "tvboptim").resolve()
    assert (result.destination / "SKILL.md").is_file()
    assert bundle_hash(result.destination) == bundle_hash(skill_source("tvboptim"))


def test_uninstall_removes_a_managed_copy(tmp_path):
    result = install(agent="codex", scope="project", project=tmp_path)
    uninstall(agent="codex", scope="project", project=tmp_path)
    assert not result.destination.exists()


def test_uninstall_preserves_modified_and_unmanaged_copies(tmp_path):
    result = install(agent="codex", scope="project", project=tmp_path)
    (result.destination / "SKILL.md").write_text("locally edited")

    with pytest.raises(SkillError, match="local modifications"):
        uninstall(agent="codex", scope="project", project=tmp_path)
    assert result.destination.exists()

    uninstall(agent="codex", scope="project", project=tmp_path, force=True)
    assert not result.destination.exists()


def test_uninstall_of_an_absent_skill_is_not_an_error(tmp_path):
    result = uninstall(agent="codex", scope="project", project=tmp_path)
    assert result.notes == ["nothing to remove"]


def test_status_reports_installed_locations(tmp_path):
    assert all(not item.exists for item in status(project=tmp_path))
    install(agent="codex", scope="project", project=tmp_path)

    found = [item for item in status(project=tmp_path) if item.exists]
    assert len(found) == 1
    assert found[0].state == "installed"


def test_project_install_never_touches_the_home_directory(tmp_path, monkeypatch):
    # A project-scoped install must not write user-scope directories, even
    # though the mapping knows about them.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    install(agent="codex", scope="project", project=tmp_path / "project")

    assert list(home.rglob("*")) == []
    assert (tmp_path / "project" / ".agents" / "skills" / "tvboptim").is_dir()


def test_cli_install_reports_its_destination(tmp_path, capsys):
    code = cli.main(
        ["skills", "install", "--agent", "claude-code", "--project", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert str(tmp_path / ".claude" / "skills" / "tvboptim") in out
    assert (tmp_path / ".claude" / "skills" / "tvboptim" / "SKILL.md").is_file()


def test_cli_dry_run_announces_without_writing(tmp_path, capsys):
    code = cli.main(
        [
            "skills",
            "install",
            "--agent",
            "codex",
            "--project",
            str(tmp_path),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Would install" in out
    assert not (tmp_path / ".agents").exists()


def test_cli_reports_errors_without_a_traceback(tmp_path, capsys):
    code = cli.main(["skills", "install", "--project", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error:")


def test_cli_export_and_status_round_trip(tmp_path, capsys):
    assert cli.main(["skills", "export", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "tvboptim" / "SKILL.md").is_file()

    capsys.readouterr()
    assert cli.main(["skills", "status", "--project", str(tmp_path)]) == 0
    assert "not installed" in capsys.readouterr().out

    cli.main(["skills", "install", "--agent", "codex", "--project", str(tmp_path)])
    capsys.readouterr()
    cli.main(["skills", "status", "--project", str(tmp_path)])
    assert "installed" in capsys.readouterr().out


def test_cli_uninstall_round_trip(tmp_path, capsys):
    cli.main(["skills", "install", "--agent", "codex", "--project", str(tmp_path)])
    target = tmp_path / ".agents" / "skills" / "tvboptim"
    assert target.is_dir()

    capsys.readouterr()
    assert (
        cli.main(
            ["skills", "uninstall", "--agent", "codex", "--project", str(tmp_path)]
        )
        == 0
    )
    assert "Removed" in capsys.readouterr().out
    assert not target.exists()


@pytest.mark.slow
def test_built_wheel_contains_the_skill_bundle(tmp_path):
    """Build the real wheel and assert the bundle survived packaging.

    Recursive globs silently drop files, so assert on the artifact rather than
    on the configuration that is supposed to produce it.
    """
    builder = shutil.which("uv")
    command = (
        [builder, "build", "--wheel", "--out-dir", str(tmp_path)]
        if builder
        else [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)]
    )
    build = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"wheel build unavailable: {build.stderr[-500:]}")

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())

    source = skill_source("tvboptim")
    expected = {
        f"tvboptim/skills/tvboptim/{path.relative_to(source).as_posix()}"
        for path in source.rglob("*")
        if path.is_file()
    }
    assert expected <= names, sorted(expected - names)
    assert "tvboptim/cli.py" in names

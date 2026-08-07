"""Command line entry point for TVB-Optim.

Currently exposes only skill installation. Kept import-light on purpose: it
pulls from :mod:`tvboptim.skills._installer`, which imports nothing from the
numerical stack.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .skills import _installer
from .skills._installer import AGENTS, SCOPES, SkillError


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by the subcommands that resolve a destination."""
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        help="agent client whose skill directory to write (selects the "
        "destination only; the copied bundle is identical for every agent)",
    )
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="project",
        help="install into the project or the user home directory (default: project)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        help="project root for --scope project (default: current directory)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="explicit target directory, bypassing the agent mapping",
    )
    parser.add_argument(
        "--skill",
        default="tvboptim",
        help="name of the bundled skill (default: tvboptim)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the destination without writing anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a directory tvboptim did not write, or one with local "
        "modifications",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tvboptim",
        description="TVB-Optim command line tools.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    skills = commands.add_parser(
        "skills",
        help="install the bundled agent skills",
        description="Copy the bundled agent skills into the directory an agent "
        "client scans.",
    )
    actions = skills.add_subparsers(dest="action", required=True)

    install = actions.add_parser("install", help="install a skill for an agent")
    _add_target_arguments(install)

    export = actions.add_parser(
        "export",
        help="copy a skill to a directory, with no agent mapping",
    )
    export.add_argument("destination", type=Path, help="directory to write into")
    export.add_argument("--skill", default="tvboptim", help="name of the bundled skill")
    export.add_argument("--dry-run", action="store_true", help="report only")
    export.add_argument("--force", action="store_true", help="replace an existing copy")

    status = actions.add_parser("status", help="show where the skill is installed")
    status.add_argument(
        "--project",
        type=Path,
        help="project root to inspect (default: current directory)",
    )
    status.add_argument("--skill", default="tvboptim", help="name of the skill")

    uninstall = actions.add_parser("uninstall", help="remove an installed skill")
    _add_target_arguments(uninstall)

    return parser


def _report(result, stream) -> None:
    """Print the destination and what happened to it."""
    verb = {
        "install": "Installed",
        "export": "Exported",
        "uninstall": "Removed",
    }[result.action]
    if result.dry_run:
        planned = {
            "install": "Would install",
            "export": "Would export",
            "uninstall": "Would remove",
        }[result.action]
        print(f"{planned} {result.skill} -> {result.destination}", file=stream)
    else:
        print(f"{verb} {result.skill} -> {result.destination}", file=stream)
    for note in result.notes:
        print(f"  note: {note}", file=stream)


def _report_status(installations, skill: str, stream) -> None:
    found = [item for item in installations if item.exists]
    if not found:
        print(f"{skill} is not installed in any known skill directory.", file=stream)
        print(
            "Run 'tvboptim skills install --agent <agent>' to install it.",
            file=stream,
        )
        return
    print(f"tvboptim {_installer.package_version()}", file=stream)
    for item in found:
        detail = item.state
        if item.manifest is not None:
            detail = f"{item.state} (from {item.manifest.package_version})"
        print(f"  {item.path}: {detail}", file=stream)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit status."""
    args = _build_parser().parse_args(argv)

    try:
        if args.action == "install":
            result = _installer.install(
                agent=args.agent,
                scope=args.scope,
                skill=args.skill,
                project=args.project,
                destination=args.destination,
                dry_run=args.dry_run,
                force=args.force,
            )
        elif args.action == "export":
            result = _installer.export(
                args.destination,
                skill=args.skill,
                dry_run=args.dry_run,
                force=args.force,
            )
        elif args.action == "uninstall":
            result = _installer.uninstall(
                agent=args.agent,
                scope=args.scope,
                skill=args.skill,
                project=args.project,
                destination=args.destination,
                dry_run=args.dry_run,
                force=args.force,
            )
        else:
            _report_status(
                _installer.status(skill=args.skill, project=args.project),
                args.skill,
                sys.stdout,
            )
            return 0
    except SkillError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _report(result, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

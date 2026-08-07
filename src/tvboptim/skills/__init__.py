"""Bundled agent skills for working with TVB-Optim.

The canonical skill sources live beside this module, one directory per skill,
and are shipped as package data. Installing them into an agent client is always
opt-in: importing or installing ``tvboptim`` never writes to an agent's
configuration on its own.

Examples
--------
>>> from pathlib import Path
>>> from tvboptim.skills import export, install
>>> install(agent="claude-code", scope="project", project=Path.cwd())  # doctest: +SKIP
>>> export(Path("./agent-skills"))  # doctest: +SKIP
"""

from ._installer import (
    AGENT_DIRECTORIES,
    AGENTS,
    SCOPES,
    Installation,
    Manifest,
    Result,
    SkillError,
    available_skills,
    export,
    install,
    resolve_destination,
    skill_source,
    status,
    uninstall,
)

__all__ = [
    "AGENTS",
    "AGENT_DIRECTORIES",
    "SCOPES",
    "Installation",
    "Manifest",
    "Result",
    "SkillError",
    "available_skills",
    "export",
    "install",
    "resolve_destination",
    "skill_source",
    "status",
    "uninstall",
]

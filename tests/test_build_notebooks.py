"""Regression checks for prose retained in downloadable notebooks."""

import runpy
from pathlib import Path

import pytest

strip_site_only_prose = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "build_notebooks.py")
)["_strip_site_only_prose"]

BUTTONS = """Try this notebook interactively:

[Download .ipynb](example.ipynb){.btn .btn-primary download="example.ipynb"}
[Download .qmd](example.qmd){.btn .btn-secondary download="example.qmd"}
[Open in Colab](https://colab.research.google.com/example){.btn .btn-warning target="_blank"}

""".replace("\n", " \n", 1)


@pytest.mark.parametrize(
    "prose",
    [
        "Introductory explanation.\n\nSee [related tutorial](related.qmd).\n\n## First section\n",
        "An introduction with no heading.\n",
        "## Introduction\n\nExisting heading-first layout.\n",
        "::: {.callout-note}\n## Updated tutorials\n\nSee [replacement](new.qmd).\n:::\n",
    ],
)
def test_strip_buttons_preserves_all_following_prose(prose):
    source = "---\ntitle: Example\njupyter: python3\n---\n\n" + BUTTONS + prose
    assert strip_site_only_prose(source) == prose


def test_page_without_buttons_keeps_ordinary_links():
    prose = "An introduction.\n\n[Download .qmd](example.qmd)\n\n## First section\n"
    assert strip_site_only_prose("---\ntitle: Example\n---\n\n" + prose) == prose

#!/usr/bin/env python3
"""Build the Colab companion notebooks that ship alongside the docs.

The rendered site is produced by ``quarto render`` from the ``.qmd`` sources.
The committed ``.ipynb`` files are a separate deliverable: they exist so the
"Download .ipynb" and "Open in Colab" buttons on each page have something to
point at. The ``.qmd`` is always the ground truth; a notebook is never edited
by hand.

``quarto convert`` alone does not produce the committed form, so this script
normalizes its output:

* the YAML frontmatter and the "Try this notebook interactively" button block
  are dropped, since neither means anything outside the rendered site;
* quarto ``#|`` cell options are stripped, as they are directives to the
  renderer rather than Python;
* the kernelspec is replaced with the canonical one. ``quarto convert`` embeds
  whichever kernel it discovers via ``python3`` on PATH, so running it from an
  activated venv bakes an absolute local path into the notebook;
* outputs are cleared and the JSON is written the way nbformat writes it.

Usage::

    python scripts/build_notebooks.py                 # rebuild every notebook
    python scripts/build_notebooks.py docs/a/b.qmd    # rebuild specific pages
    python scripts/build_notebooks.py --check         # report staleness only
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

# quarto convert reports the kernel it resolved, including an absolute path to
# it. Only the portable fields belong in a committed notebook.
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}

FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
BUTTON_BLOCK = re.compile(
    r"\A\s*Try this notebook interactively:.*?(?=^#)", re.S | re.M
)


def notebook_pages():
    """Every doc page that advertises a downloadable notebook."""
    return sorted(
        qmd
        for qmd in DOCS.rglob("*.qmd")
        if "_freeze" not in qmd.parts
        and "Try this notebook interactively" in qmd.read_text()
    )


def _strip_site_only_prose(text: str) -> str:
    return BUTTON_BLOCK.sub("", FRONTMATTER.sub("", text)).lstrip("\n")


def _normalize(notebook: dict) -> dict:
    notebook["metadata"] = {"kernelspec": KERNELSPEC}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = [
                line for line in cell["source"] if not line.lstrip().startswith("#|")
            ]
            cell["outputs"] = []
            cell["execution_count"] = None
    return notebook


def render(qmd: pathlib.Path) -> str:
    """Convert one page and return the notebook JSON, without writing it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        staged = pathlib.Path(tmpdir) / qmd.name
        staged.write_text(_strip_site_only_prose(qmd.read_text()))
        subprocess.run(
            ["quarto", "convert", str(staged)], check=True, capture_output=True
        )
        notebook = json.loads(staged.with_suffix(".ipynb").read_text())
    # nbformat omits the trailing newline, and quarto indents by one space.
    return json.dumps(_normalize(notebook), indent=1, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="*", type=pathlib.Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report notebooks that are out of date instead of rewriting them",
    )
    args = parser.parse_args()

    pages = args.pages or notebook_pages()
    stale = []
    for qmd in pages:
        target = qmd.with_suffix(".ipynb")
        built = render(qmd)
        current = target.read_text() if target.exists() else None
        if built == current:
            continue
        stale.append(target)
        if args.check:
            print(f"out of date: {target}")
        else:
            target.write_text(built)
            print(f"{'updated' if current else 'created'} {target}")

    if args.check and stale:
        print(f"\n{len(stale)} notebook(s) out of date; run {sys.argv[0]}")
        return 1
    if not stale:
        print("all notebooks up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

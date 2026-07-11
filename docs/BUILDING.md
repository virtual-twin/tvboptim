# Building the documentation

The site is a [Quarto](https://quarto.org) website under `docs/`. Most pages are
notebooks (`.qmd`) that run real simulations. To keep CI fast and reproducible, CI does
**not** execute those notebooks. It renders them from committed **freeze** outputs
instead. Executing notebooks is a local step you run when you change their code.

## Prerequisites

- [Quarto CLI](https://quarto.org/docs/get-started/)
- Python env synced from the lockfile: `uv sync --all-extras`

## The freeze contract

`docs/_quarto.yml` sets `execute: freeze: auto`. Quarto stores each notebook's executed
outputs (figures plus a results JSON) under `docs/_freeze/`, which is committed to git.

- When a notebook's `.qmd` source is unchanged, `quarto render` reuses the frozen
  outputs and never starts a Python kernel. This is what CI does.
- When you change a notebook's code, re-render it locally so its `docs/_freeze/` entry
  is regenerated, then commit that entry alongside the `.qmd`. If you forget, CI falls
  back to executing that one notebook, which is slow and depends on the CI environment.

This replaces the previous scheme of committing `@cache` pickles. Those pickles stored
live JAX/equinox objects and broke whenever a library upgrade moved an internal type.
Freeze stores rendered outputs, so it is immune to that version skew.

## Local build

Run from the repository root.

1. Sync the environment (matches CI's pinned deps):

   ```bash
   uv sync --all-extras
   ```

2. Build the API reference:

   ```bash
   uv run quartodoc build --config docs/_quarto.yml
   ```

3. Render the site. Notebooks whose source changed are executed and their
   `docs/_freeze/` entries refreshed; the rest render from freeze:

   ```bash
   uv run quarto render docs/
   ```

   Preview with live reload instead:

   ```bash
   uv run quarto preview docs/
   ```

4. Refresh the Colab notebooks (`.qmd` to `.ipynb`), only when a workflow changed:

   ```bash
   ./docs/convert-workflows.sh
   ```

5. Commit any changed `docs/_freeze/` entries together with the `.qmd` (and `.ipynb`)
   you edited. The rendered `docs/_site/` is a build artifact and is not committed.

## What CI does

The `docs.yml` workflow runs `quartodoc build` then `quarto render docs/` with no
special compute. Because every notebook is frozen, no Python kernel runs during render.
CI serves whatever is in the committed `docs/_freeze/`, so keeping freeze up to date is
the contributor's responsibility (step 5 above).

## Caching notes

The `@cache` decorator (`src/tvboptim/utils/caching.py`) is a **local authoring**
convenience that pickles a function's result under `docs/**/cache/`. Those pickles are
gitignored, not committed. The loader is defensive: if a pickle fails to load, or its
recorded jax/tvboptim version or the function source has changed, it warns and
recomputes instead of failing. If you want to force a rebuild, pass `redo=True` or
delete the cache directory.

If a render ever dies on an old committed pickle (for example
`ModuleNotFoundError: No module named 'jax._src.prng'`), the fix is to regenerate the
notebook's freeze under the current environment (step 3) and commit it.

#!/bin/bash
# Convert workflow .qmd files to .ipynb notebooks
# This runs after `quarto render` to create downloadable notebooks

set -e  # Exit on error

echo "Converting workflow notebooks..."

# Ensure output directory exists
mkdir -p docs/_site/workflows/

# Convert each workflow qmd file to ipynb
for qmd_file in docs/workflows/*.qmd; do
    if [ -f "$qmd_file" ]; then
        filename=$(basename "$qmd_file" .qmd)
        echo "  Converting ${filename}.qmd..."

        # Convert directly to _site directory
        quarto convert "$qmd_file" --output "docs/_site/workflows/${filename}.ipynb"
    fi
done

echo "Conversion complete!"
echo "Notebooks available in docs/_site/workflows/"

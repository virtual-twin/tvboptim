#!/bin/bash
# Convert .qmd files to .ipynb notebooks
# This runs after `quarto render` to create downloadable notebooks

set -e  # Exit on error

echo "Converting notebooks..."

# Define directories to process
DIRS=("workflows" "advanced")

# Process each directory
for dir in "${DIRS[@]}"; do
    echo "Processing docs/${dir}..."

    # Ensure output directory exists
    mkdir -p "docs/_site/${dir}/"

    # Convert each qmd file to ipynb
    for qmd_file in docs/${dir}/*.qmd; do
        if [ -f "$qmd_file" ]; then
            filename=$(basename "$qmd_file" .qmd)
            echo "  Converting ${filename}.qmd..."

            # Convert directly to _site directory
            quarto convert "$qmd_file" --output "docs/_site/${dir}/${filename}.ipynb"
        fi
    done
done

echo "Conversion complete!"
echo "Notebooks available in docs/_site/workflows/ and docs/_site/advanced/"

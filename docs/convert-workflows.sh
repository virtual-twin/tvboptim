#!/bin/bash
# Convert .qmd files to .ipynb notebooks in source directories
# This creates Colab-friendly notebooks that get committed to the repo

set -e  # Exit on error

echo "Converting .qmd files to .ipynb notebooks..."

# Define directories to process (add more as needed)
DIRS=("workflows" "advanced")

# Process each directory
for dir in "${DIRS[@]}"; do
    docs_dir="docs/${dir}"

    # Skip if directory doesn't exist
    if [ ! -d "$docs_dir" ]; then
        echo "  Skipping ${docs_dir} (directory not found)"
        continue
    fi

    echo "Processing ${docs_dir}..."

    # Convert each qmd file to ipynb in the same directory
    qmd_files=("${docs_dir}"/*.qmd)
    if [ ! -e "${qmd_files[0]}" ]; then
        echo "  No .qmd files found in ${docs_dir}"
        continue
    fi

    for qmd_file in "${qmd_files[@]}"; do
        if [ -f "$qmd_file" ]; then
            filename=$(basename "$qmd_file" .qmd)
            ipynb_file="${docs_dir}/${filename}.ipynb"

            echo "  Converting ${filename}.qmd → ${filename}.ipynb"

            # Convert to notebook in same directory
            quarto convert "$qmd_file" --output "$ipynb_file"
        fi
    done
done

echo ""
echo "Cleaning notebooks for Google Colab..."
uv run python docs/clean-notebooks.py

echo ""
echo "✓ Conversion complete!"
echo "  Notebooks are now ready for:"
echo "  - Committing to git (for Colab integration)"
echo "  - Opening in Google Colab via GitHub"

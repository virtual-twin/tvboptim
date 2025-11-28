#!/bin/bash
# Build documentation locally
# This simulates the CI workflow and builds the complete documentation site

set -e  # Exit on error

echo "========================================"
echo "Building Documentation"
echo "========================================"
echo

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from repository root"
    exit 1
fi

# Step 1: Convert notebooks (before rendering)
echo "Step 1: Converting .qmd to .ipynb notebooks..."
./docs/convert-workflows.sh
echo "✓ Notebooks converted and cleaned"
echo

# Step 2: Build API reference
echo "Step 2: Building API reference with quartodoc..."
uv run quartodoc build --config docs/_quarto.yml
echo "✓ API reference built"
echo

# Step 3: Render documentation
echo "Step 3: Rendering Quarto documentation..."
quarto render docs/
echo "✓ Documentation rendered (.ipynb files ignored)"
echo

# Step 4: Check outputs
echo "Step 4: Verifying outputs..."

if [ ! -d "docs/_site" ]; then
    echo "✗ Error: docs/_site directory not created"
    exit 1
fi

# Check for HTML files
html_count=$(find docs/_site -name "*.html" | wc -l)
echo "  Found $html_count HTML files"

# Check for converted notebooks in source directories
ipynb_count=$(find docs/workflows docs/advanced -name "*.ipynb" 2>/dev/null | wc -l)
echo "  Found $ipynb_count notebook files in source directories"

if [ "$ipynb_count" -eq 0 ]; then
    echo "  ⚠ Warning: No .ipynb files found in source directories"
    echo "  This might indicate conversion failed"
fi

echo
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo
echo "All build steps completed successfully!"
echo
echo "To preview the site locally, run:"
echo "  quarto preview docs/"
echo
echo "Or to rebuild everything, run this script again:"
echo "  ./docs/build_docs.sh"
echo

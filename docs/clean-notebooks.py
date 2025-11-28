#!/usr/bin/env python3
"""
Clean Jupyter notebooks for Google Colab by removing Quarto metadata.

This script removes:
1. YAML front matter from the first markdown cell
2. Quarto cell directives (lines starting with #|)
3. Quarto-specific markdown syntax
"""

import json
import sys
from pathlib import Path
import re


def remove_yaml_frontmatter(cell_source):
    """Remove YAML front matter from markdown cell."""
    if not cell_source:
        return cell_source

    # Track original format
    was_list = isinstance(cell_source, list)

    # Join lines if it's a list (preserving newlines)
    if was_list:
        content = ''.join(cell_source)
    else:
        content = cell_source

    # Check if it starts with YAML front matter
    if content.strip().startswith('---'):
        # Find the closing ---
        lines = content.split('\n')
        yaml_end = -1
        for i, line in enumerate(lines[1:], 1):  # Start from second line
            if line.strip() == '---':
                yaml_end = i
                break

        if yaml_end > 0:
            # Remove YAML front matter and any following empty lines
            remaining_lines = lines[yaml_end + 1:]

            # Remove leading empty lines
            while remaining_lines and not remaining_lines[0].strip():
                remaining_lines.pop(0)

            # Return in proper format with newlines
            if was_list:
                # Each line should end with \n except possibly the last
                return [line + '\n' for line in remaining_lines[:-1]] + ([remaining_lines[-1]] if remaining_lines else [])
            else:
                return '\n'.join(remaining_lines)

    return cell_source


def remove_quarto_directives(cell_source):
    """Remove Quarto cell directives from code cells."""
    if not cell_source:
        return cell_source

    # Track original format
    was_list = isinstance(cell_source, list)

    # Handle both list and string formats
    if was_list:
        # Preserve newlines - join to process, then split back
        content = ''.join(cell_source)
        lines = content.split('\n')
    else:
        lines = cell_source.split('\n')

    # Filter out lines starting with #|
    cleaned_lines = [line for line in lines if not line.strip().startswith('#|')]

    # Return in proper format with newlines
    if was_list:
        if not cleaned_lines:
            return []
        # Each line should end with \n except possibly the last
        return [line + '\n' for line in cleaned_lines[:-1]] + ([cleaned_lines[-1]] if cleaned_lines else [])
    return '\n'.join(cleaned_lines)


def clean_quarto_markdown(cell_source):
    """Clean Quarto-specific markdown syntax."""
    if not cell_source:
        return cell_source

    # Track original format
    was_list = isinstance(cell_source, list)

    # Join lines if it's a list (preserving newlines)
    if was_list:
        content = ''.join(cell_source)
    else:
        content = cell_source

    # Remove Quarto callout blocks (:::{.callout-note} etc)
    content = re.sub(r':::\{[^}]+\}.*?\n', '', content)
    content = re.sub(r'^\s*:::\s*$', '', content, flags=re.MULTILINE)

    # Clean up Quarto button syntax
    # Remove text-based Colab button (new format)
    content = re.sub(r'\[Open in Colab\]\([^)]+\)\{[^}]+\}\s*\n*', '', content)
    # Remove Colab badge (old format)
    content = re.sub(r'\[!\[Open In Colab\]\([^)]+\)\]\([^)]+\)\s*\n*', '', content)
    # Remove download buttons
    content = re.sub(r'\[Download \.qmd\]\([^)]+\)\{[^}]+\}\s*\n*', '', content)
    content = re.sub(r'\[Download \.ipynb\]\([^)]+\)\{[^}]+\}\s*\n*', '', content)
    content = re.sub(r'<span class="btn[^>]*>.*?</span>\s*\n*', '', content, flags=re.DOTALL)

    # Clean up image syntax with attributes
    # ![caption](path){#id}  becomes  ![caption](path)
    content = re.sub(r'(\!\[[^\]]*\]\([^)]+\))\{[^}]+\}', r'\1', content)

    # Remove "Try this notebook interactively:" header completely
    content = re.sub(r'Try this notebook interactively:\s*\n*', '', content)

    # Remove leading/trailing whitespace but preserve internal structure
    content = content.strip()

    # Return in proper format with newlines
    if was_list:
        if not content:
            return []
        lines = content.split('\n')
        # Each line should end with \n except possibly the last
        return [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines else [])
    return content


def clean_notebook(notebook_path):
    """Clean a single notebook file."""
    print(f"Cleaning {notebook_path}...")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Process cells
    cells_to_process = []
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])

        if cell_type == 'markdown':
            # Remove YAML front matter from first markdown cell
            if i == 0:
                source = remove_yaml_frontmatter(source)

            # Clean Quarto markdown syntax
            source = clean_quarto_markdown(source)
            cell['source'] = source
            cells_to_process.append(cell)

        elif cell_type == 'code':
            # Check if this cell has #| eval: false (meant for display only)
            source_text = ''.join(source) if isinstance(source, list) else source

            if '#| eval: false' in source_text or '#|eval:false' in source_text:
                # Convert to markdown cell with code formatting
                cleaned_source = remove_quarto_directives(source)
                code_text = ''.join(cleaned_source) if isinstance(cleaned_source, list) else cleaned_source
                code_text = code_text.strip()

                # Create markdown cell with code block
                markdown_source = f'```python\n{code_text}\n```'
                lines = markdown_source.split('\n')
                markdown_list = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines else [])

                cell['cell_type'] = 'markdown'
                cell['source'] = markdown_list
                # Remove code cell specific fields
                cell.pop('execution_count', None)
                cell.pop('outputs', None)
                cells_to_process.append(cell)
            else:
                # Regular code cell - just remove directives
                source = remove_quarto_directives(source)
                cell['source'] = source
                cells_to_process.append(cell)
        else:
            cells_to_process.append(cell)

    nb['cells'] = cells_to_process

    # Write cleaned notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  ✓ Cleaned {notebook_path}")


def main():
    """Clean all notebooks in specified directories."""
    # Directories to process (source directories, not _site)
    dirs_to_process = ['docs/workflows/', 'docs/advanced/']

    for dir_path in dirs_to_process:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"Directory {dir_path} does not exist, skipping...")
            continue

        # Find all .ipynb files
        notebooks = list(dir_path.glob('*.ipynb'))

        if not notebooks:
            print(f"No notebooks found in {dir_path}")
            continue

        print(f"\nProcessing {len(notebooks)} notebooks in {dir_path}...")
        for nb_path in notebooks:
            clean_notebook(nb_path)

    print("\n✓ All notebooks cleaned!")


if __name__ == '__main__':
    main()

#!/bin/bash
# Build script for gsply
set -e

echo "=========================================="
echo "Building gsply"
echo "=========================================="

# Clean old builds
echo "Cleaning old builds..."
rm -rf build/ dist/ *.egg-info

# Build wheel and sdist
echo "Building wheel and source distribution..."
python -m build

# Check distribution
echo "Checking distribution..."
python -m twine check dist/*

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
ls -lh dist/
echo ""
echo "To install locally:"
echo "  pip install dist/*.whl"
echo ""
echo "To upload to Test PyPI:"
echo "  python -m twine upload --repository testpypi dist/*"
echo ""
echo "To upload to PyPI:"
echo "  python -m twine upload dist/*"
echo ""

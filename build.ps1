# Build script for gsply (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building gsply" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Clean old builds
Write-Host "Cleaning old builds..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
Get-ChildItem -Filter "*.egg-info" -Recurse | Remove-Item -Recurse -Force

# Build wheel and sdist
Write-Host "Building wheel and source distribution..." -ForegroundColor Yellow
python -m build

# Check distribution
Write-Host "Checking distribution..." -ForegroundColor Yellow
python -m twine check dist/*

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Build complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Get-ChildItem "dist"
Write-Host ""
Write-Host "To install locally:" -ForegroundColor Cyan
Write-Host "  pip install dist/*.whl"
Write-Host ""
Write-Host "To upload to Test PyPI:" -ForegroundColor Cyan
Write-Host "  python -m twine upload --repository testpypi dist/*"
Write-Host ""
Write-Host "To upload to PyPI:" -ForegroundColor Cyan
Write-Host "  python -m twine upload dist/*"
Write-Host ""

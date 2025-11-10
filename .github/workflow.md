# GitHub Configuration

This directory contains GitHub-specific configuration files for the gsply project.

## CI/CD Workflows

### test.yml
- Runs on: Push to main/develop, Pull Requests
- Tests on: Python 3.10, 3.11, 3.12, 3.13
- Platforms: Ubuntu, Windows, macOS
- Features:
  - Unit tests with pytest
  - Coverage reporting to Codecov
  - Code linting with ruff
  - Type checking with mypy

### build.yml
- Runs on: Push to main/develop, Pull Requests, Tags
- Builds: Source distribution and universal wheel
- Verifies: Installation on all platforms
- Artifacts: Uploaded for download

### publish.yml
- Runs on: GitHub Release creation
- Publishes to: PyPI and TestPyPI
- Uses: Trusted publishing (no API tokens needed)
- Creates: Signed artifacts with Sigstore
- Uploads: Artifacts to GitHub Release

### benchmark.yml
- Runs on: Push to main, Pull Requests, Manual trigger
- Benchmarks: Read/write performance
- Compares: Against plyfile and Open3D
- Reports: Results as PR comment

### docs.yml
- Runs on: Push to main, Pull Requests
- Checks: README, BUILD.md, LICENSE existence
- Validates: Markdown formatting
- Verifies: API docstrings

## Issue Templates

- **bug_report.md**: Template for reporting bugs
- **feature_request.md**: Template for requesting features

## Other Files

- **CODEOWNERS**: Defines code ownership for auto-review assignments
- **dependabot.yml**: Automated dependency updates
- **pull_request_template.md**: Template for pull requests
- **CONTRIBUTING.md**: Guidelines for contributors

## Setting Up CI/CD

### For Testing & Building
No setup needed! Workflows run automatically on push/PR.

### For Publishing to PyPI

1. Go to repository Settings > Environments
2. Create environment named "pypi"
3. Add protection rules (require review for production)
4. Configure Trusted Publishing on PyPI:
   - Go to https://pypi.org/manage/account/publishing/
   - Add GitHub as publisher
   - Repository: OpsiClear/gsply
   - Workflow: publish.yml
   - Environment: pypi

### For Codecov Integration

1. Go to https://codecov.io
2. Connect your GitHub repository
3. Add `CODECOV_TOKEN` to repository secrets (if needed)

### Creating a Release

1. Update version in code
2. Commit and push to main
3. Create a new release on GitHub
4. Add tag: v0.1.0 (for example)
5. Write release notes
6. Publish release
7. CI/CD will automatically build and publish to PyPI

## Badges for README

Add these to your README.md:

```markdown
[![Tests](https://github.com/OpsiClear/gsply/workflows/Test/badge.svg)](https://github.com/OpsiClear/gsply/actions/workflows/test.yml)
[![Build](https://github.com/OpsiClear/gsply/workflows/Build/badge.svg)](https://github.com/OpsiClear/gsply/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/OpsiClear/gsply/branch/main/graph/badge.svg)](https://codecov.io/gh/OpsiClear/gsply)
[![PyPI version](https://badge.fury.io/py/gsply.svg)](https://badge.fury.io/py/gsply)
[![Python Versions](https://img.shields.io/pypi/pyversions/gsply.svg)](https://pypi.org/project/gsply/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

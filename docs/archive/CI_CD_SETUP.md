# CI/CD Setup Complete

gsply now has a complete GitHub Actions CI/CD pipeline!

## What Was Added

### GitHub Actions Workflows (`.github/workflows/`)

1. **test.yml** - Automated Testing
   - Runs on: Every push and PR to main/develop
   - Tests: Python 3.10, 3.11, 3.12, 3.13
   - Platforms: Ubuntu, Windows, macOS
   - Features:
     - Unit tests with pytest
     - Code coverage with Codecov
     - Linting with ruff
     - Type checking with mypy

2. **build.yml** - Build & Distribution
   - Runs on: Push, PR, and tags
   - Builds: Universal wheel + source distribution
   - Verifies: Installation on all platforms
   - Artifacts: Uploaded for download

3. **publish.yml** - PyPI Publishing
   - Runs on: GitHub Release creation
   - Publishes to: PyPI (production) and TestPyPI
   - Security: Trusted publishing (no API tokens)
   - Features: Sigstore artifact signing

4. **benchmark.yml** - Performance Testing
   - Runs on: Push to main, PRs, manual trigger
   - Benchmarks: Read/write performance
   - Compares: Against plyfile and Open3D
   - Reports: Results as PR comments

5. **docs.yml** - Documentation Validation
   - Runs on: Push and PR
   - Checks: README, docs/BUILD.md, LICENSE
   - Validates: Markdown formatting
   - Verifies: API docstrings

### GitHub Templates

1. **Issue Templates** (`.github/ISSUE_TEMPLATE/`)
   - `bug_report.md` - Bug reporting template
   - `feature_request.md` - Feature request template

2. **Pull Request Template** (`.github/`)
   - `pull_request_template.md` - PR checklist

### Configuration Files

1. **CODEOWNERS** - Auto-assign reviewers
2. **dependabot.yml** - Automated dependency updates
3. **CONTRIBUTING.md** - Contribution guidelines
4. **README.md** - GitHub config documentation

## Quick Start

### For Contributors

The CI/CD will run automatically on:
- Every push to main/develop branches
- Every pull request
- Creation of tags (v*)
- GitHub releases

No manual action needed!

### For Maintainers

#### Running Tests Locally

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=gsply

# Run benchmarks
pip install -e ".[benchmark]"
python benchmark.py
```

#### Creating a Release

1. Update version:
   ```bash
   # Edit pyproject.toml
   version = "0.2.0"

   # Edit src/gsply/__init__.py
   __version__ = "0.2.0"
   ```

2. Update release notes:
   ```bash
   # Edit docs/RELEASE_NOTES.md
   ```

3. Commit and tag:
   ```bash
   git add .
   git commit -m "Release v0.2.0"
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push && git push --tags
   ```

4. Create GitHub Release:
   - Go to GitHub > Releases > Create new release
   - Choose tag: v0.2.0
   - Write release notes
   - Publish release

5. CI/CD will automatically:
   - Build wheels
   - Run tests
   - Publish to TestPyPI
   - Publish to PyPI (if configured)
   - Upload artifacts to GitHub Release

## Setting Up PyPI Publishing

### Step 1: PyPI Trusted Publishing

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new publisher"
3. Fill in:
   - PyPI Project Name: `gsply`
   - Owner: `OpsiClear`
   - Repository: `gsply`
   - Workflow: `publish.yml`
   - Environment: `pypi`

### Step 2: GitHub Environment

1. Go to GitHub > Settings > Environments
2. Create environment: `pypi`
3. Add protection rules:
   - ✓ Required reviewers (optional but recommended)
   - ✓ Wait timer (optional)

### Step 3: TestPyPI (Optional)

Repeat Step 1 for TestPyPI:
- Go to https://test.pypi.org/manage/account/publishing/
- Same settings as above

## Setting Up Codecov

1. Go to https://codecov.io
2. Sign in with GitHub
3. Add repository: `OpsiClear/gsply`
4. Copy the token (if needed)
5. Add to GitHub Secrets:
   - Settings > Secrets > Actions
   - New secret: `CODECOV_TOKEN`

## CI/CD Features

### Automated Testing

- ✅ Multi-platform (Linux, Windows, macOS)
- ✅ Multi-version (Python 3.10-3.13)
- ✅ Code coverage tracking
- ✅ Lint checking (ruff)
- ✅ Type checking (mypy)

### Automated Building

- ✅ Universal wheel (`py3-none-any`)
- ✅ Source distribution
- ✅ Installation verification
- ✅ Artifact storage

### Automated Publishing

- ✅ PyPI publication on release
- ✅ TestPyPI for testing
- ✅ Trusted publishing (secure)
- ✅ Sigstore signing
- ✅ GitHub Release artifacts

### Quality Checks

- ✅ Documentation validation
- ✅ Performance benchmarking
- ✅ Dependency updates (Dependabot)
- ✅ Code ownership (auto-review)

## Workflow Triggers

| Workflow | Push | PR | Tag | Release | Manual |
|----------|------|-----|-----|---------|--------|
| test.yml | ✓ | ✓ | - | - | - |
| build.yml | ✓ | ✓ | ✓ | - | - |
| publish.yml | - | - | - | ✓ | - |
| benchmark.yml | ✓ | ✓ | - | - | ✓ |
| docs.yml | ✓ | ✓ | - | - | - |

## Status Badges

Add these to your README.md:

```markdown
[![Tests](https://github.com/OpsiClear/gsply/workflows/Test/badge.svg)](https://github.com/OpsiClear/gsply/actions/workflows/test.yml)
[![Build](https://github.com/OpsiClear/gsply/workflows/Build/badge.svg)](https://github.com/OpsiClear/gsply/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/OpsiClear/gsply/branch/main/graph/badge.svg)](https://codecov.io/gh/OpsiClear/gsply)
[![PyPI version](https://badge.fury.io/py/gsply.svg)](https://badge.fury.io/py/gsply)
[![Python Versions](https://img.shields.io/pypi/pyversions/gsply.svg)](https://pypi.org/project/gsply/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

## Testing the Workflows

### Local Testing

```bash
# Install act (GitHub Actions local runner)
# https://github.com/nektos/act

# Run test workflow locally
act -j test

# Run build workflow locally
act -j build
```

### First Push

After pushing to GitHub:

1. Go to Actions tab
2. Watch workflows run
3. Check for any errors
4. Fix any configuration issues

## Troubleshooting

### Workflow fails with "No module named 'gsply'"

- Check that `pip install -e .` is in the workflow
- Verify `pyproject.toml` is correct

### PyPI publishing fails

- Verify trusted publishing is configured
- Check GitHub environment name matches
- Ensure version number is updated

### Tests fail on Windows

- Check for path separator issues (`/` vs `\`)
- Use `Path` from `pathlib` for cross-platform paths

### Coverage not uploading

- Check `CODECOV_TOKEN` secret is set
- Verify repository is added to Codecov

## Next Steps

1. **Push to GitHub**: All workflows will run automatically
2. **Review Results**: Check Actions tab for status
3. **Configure PyPI**: Set up trusted publishing
4. **Add Badges**: Update README.md with status badges
5. **Create Release**: Test the full publishing pipeline

## Documentation

Full documentation available in:
- `.github/README.md` - GitHub config overview
- `.github/CONTRIBUTING.md` - Contribution guidelines
- `docs/BUILD.md` - Build instructions
- `docs/RELEASE_NOTES.md` - Release history

---

**CI/CD pipeline successfully configured!** gsply now has enterprise-grade automation.

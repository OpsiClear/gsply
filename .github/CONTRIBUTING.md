# Contributing to gsply

Thank you for your interest in contributing to gsply! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gsply.git
   cd gsply
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev,benchmark]"
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

4. Run benchmarks (if performance-related):
   ```bash
   python benchmark.py
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request

## Code Style

- Use Python 3.10+ type hints
- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add docstrings to all public functions
- Keep functions focused and concise

## Testing

- Write tests for all new features
- Ensure all existing tests pass
- Aim for high test coverage
- Test on multiple Python versions (3.10, 3.11, 3.12, 3.13)

## Performance

- Benchmark performance-critical changes
- Avoid unnecessary allocations
- Use numpy operations efficiently
- Document performance improvements

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update RELEASE_NOTES.md

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to RELEASE_NOTES.md
4. Fill out the PR template completely
5. Request review

## Performance Benchmarks

Before submitting performance improvements:

1. Run the benchmark suite:
   ```bash
   python benchmark.py
   ```

2. Document the improvement:
   - Before/after timings
   - Test data size
   - System specs

3. Include results in PR description

## Code Review

- Be respectful and constructive
- Respond to feedback promptly
- Make requested changes in new commits
- Squash commits before merge (if requested)

## Release Process

Only maintainers can create releases:

1. Update version in `pyproject.toml` and `src/gsply/__init__.py`
2. Update `RELEASE_NOTES.md`
3. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push --tags`
5. GitHub Actions will build and publish to PyPI

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

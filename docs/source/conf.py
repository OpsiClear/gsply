# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# Read the Docs path setup
if os.environ.get("READTHEDOCS") == "True":
    # RTD uses a different directory structure
    # The repository root is typically one level up from docs/source
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))
else:
    # Local development path
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "gsply"
author = "OpsiClear"
copyright = f"{datetime.now().year}, {author}"
version = "0.2.6"
release = "0.2.6"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
napoleon_use_param = True
napoleon_use_rtype = False
todo_include_todos = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "gsply Documentation"
html_short_title = "gsply"
html_logo = None
html_favicon = None

html_theme_options = {
    "announcement": None,
    "sidebar_hide_name": False,
    "light_logo": None,
    "dark_logo": None,
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1e40af",
        "color-admonition-background": "#f0f9ff",
        "font-stack": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
        "color-admonition-background": "#1e3a5f",
        "font-stack": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace",
    },
    "navigation_with_keys": True,
    "source_repository": "https://github.com/OpsiClear/gsply",
    "source_branch": "master",
    "source_directory": "docs/source/",
}

html_css_files = [
    "custom.css",
]

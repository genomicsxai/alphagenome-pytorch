# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the source directory to the path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'AlphaGenome PyTorch'
copyright = '2026, Kundaje Lab'
author = 'Kundaje Lab'

# Get version from installed package metadata (avoids importing torch)
from importlib.metadata import version as get_version, PackageNotFoundError
try:
    release = get_version("alphagenome-pytorch")
except PackageNotFoundError:
    release = "0.0.0.dev0"

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/genomicsxai/alphagenome-pytorch',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
    'article_header_start': ['toggle-primary-sidebar.html', 'breadcrumbs'],
}
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
# Render `Attributes:` sections as inline :ivar: fields. Without this, napoleon
# emits a separate py:attribute for each entry, which collides with the ones
# autodoc already generates from `undoc-members` (e.g. the GeneCounts dataclass
# fields) and warns about duplicate object descriptions. This is global: it also
# changes how any future page documenting a class with an `Attributes:` docstring
# section renders. As of this change only api/aggregation is affected.
napoleon_use_ivar = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# Mock heavy imports so docs build without torch installed
autodoc_mock_imports = ['torch']

# NOTE: `torch` being mocked means `torch.Tensor | None` raises TypeError if it is
# ever *evaluated* — mock objects don't implement `|`. Signature annotations are
# evaluated at def time unless the module opts into PEP 563, so every module
# reachable from the package __init__ needs `from __future__ import annotations`.
# Autodoc reimports the package under the mock (with try_reload=True), so a module
# missing it takes down every autodoc page, not just its own.

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'

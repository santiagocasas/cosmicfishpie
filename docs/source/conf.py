# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath("../../"))

from cosmicfishpie import VERSION, VERSION_SHORT  # noqa: E402

# -- Project information -----------------------------------------------------

project = "cosmicfishpie"
copyright = f"{datetime.today().year}, RWTH Aachen University"
author = "S. Casas, M. Martinelli, S. Pamuk, Sabarish V.M."
version = VERSION_SHORT
release = VERSION


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

# Tell myst-parser to assign header anchors for h1-h3.
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "deflist"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

source_suffix = [".rst", ".md"]

# GitHub renders README anchors client-side, so the anchor checker cannot see them.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/.*"]

modindex_common_prefix = ["cosmicfishpie."]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "getdist": ("https://getdist.readthedocs.io/en/latest/", None),
}

# Keep members in source order, as the authors arranged them.
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_numpy_docstring = True

# Include default values when documenting parameter types.
typehints_defaults = "comma"

napoleon_preprocess_types = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_title = f"cosmicfishpie v{VERSION}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/santiagocasas/cosmicfishpie",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cosmicfishpie/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "back_to_top_button": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_context = {
    "default_mode": "auto",
    "github_user": "santiagocasas",
    "github_repo": "cosmicfishpie",
    "github_version": "main",
    "doc_path": "docs/source",
}

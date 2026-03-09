# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'dashi'
copyright = '2026, Carlos Sáez Silvestre, David Fernández Narro, Pablo Ferri Borredá, Ángel Sánchez García'
author = 'Carlos Sáez Silvestre, David Fernández Narro, Pablo Ferri Borredá, Ángel Sánchez García'
release = '0.3.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None  # Add a logo file path here if you have one
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
    'titles_only': False,
}

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings --------------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

add_module_names = False
autosummary_generate = True
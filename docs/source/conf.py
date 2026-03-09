# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))     

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dashi'
copyright = '2024, Carlos Sáez Silvestre, David Fernández Narro, Pablo Ferri Borredá, Ángel Sánchez García'
author = 'Carlos Sáez Silvestre, David Fernández Narro, Pablo Ferri Borredá, Ángel Sánchez García'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx_autodoc_typehints',  # Automatically document type hints
]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

napoleon_google_docstring = False  # Disable Google-style
napoleon_numpy_docstring = True  # Enable NumPy-style
napoleon_include_init_with_doc = True  # Include __init__ method docstrings
napoleon_use_param = True  # Use :param: syntax in output
napoleon_use_rtype = True  # Use :rtype: syntax in output

autodoc_default_options = {                        # Fixed: was "utodoc_default_options"
    'members': True,
    'undoc-members': True,  # Ensure undocumented members are also listed
    'show-inheritance': True,
}

add_module_names = False

autosummary_generate = True  # Auto-generate summaries

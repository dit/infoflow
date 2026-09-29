import infoflow

project = "infoflow"
copyright = "2026, dit contributors"  # noqa: A001
version = release = infoflow.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
master_doc = "index"
exclude_patterns = ["_build"]
add_module_names = False
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "dit": ("https://dit.readthedocs.io/en/latest", None),
}
html_theme = "sphinx_rtd_theme"
napoleon_use_ivar = True

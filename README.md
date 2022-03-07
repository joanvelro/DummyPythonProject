# Dummy Project

```
├── docs                    # auto-documentation (latex, html, pdf)
│ 
├── src                     # source code
│   ├── __init__.py         # make it a package
│   └── main.py             # main program
│   └── utils.py            # commom functionalities
└── test
│    ├── __init__.py        # also make test a package
│    └── test.py            # unit tests
│
├── requirements.txt        # python dependencies 
├── README.md               # this document
```


# Sphinx Tutorial
*  Install Sphinx and Rinohtype in the virtual environment of the project you’re working on use the following commands below.
```
conda activate env_name
pip install Sphinx
pip install rinohtype
```
*  Create a docs directory and cd into this directory.
```
mkdir docs
cd docs
```
* Setup Sphinx
```
sphinx-quickstart
```
* Open source/conf.py
* Configure path to root directory
* Add extensions
```
[ 'rinoh.frontend.sphinx', 'sphinx.ext.viewcode', 'sphinx.ext.todo', 'sphinx.ext.autodoc']
```
* Open the index.rst and change the content to the following. (Click the index.rst  link for full content)
```
Documentation for module
**************************
.. toctree::
   :maxdepth: 2
   :caption: Contents:

Module
===================
.. automodule:: src
   :members:


utils
===================
.. automodule:: src.utils
   :members:

main
===================
.. automodule:: src.main
   :members:


Documentation for testing
**************************
.. toctree::
   :maxdepth: 2
   :caption: Test Contents:

Test
===================
.. automodule:: test.test
   :members:

```

* where automodule correspond with the name of the python file 
```
"""
.. module:: src.main
   :synopsis: define the functionality

.. moduleauthor:: (C) Jose Angel Velasco - 2021
"""
```

* Still inside the docs directory run, Create the HTML documentation files.
```
make html
```
* Create the latex documentation files
```
make latex
```
* Create the PDF documentation files
```
sphinx-build -b rinoh source _build/rinoh
```

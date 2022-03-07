# Dummy project - Crimes 
## Predictive modelling

* Detect the relationship between the crimes and the different features and characteristics availables


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

* 'Other',no Violent Crime
* 'Confidence Games'no Violent Crime
* 'Larceny',no Violent Crime
  'Auto Theft',no Violent Crime
* 'Residential Burglary',no Violent Crime
* 'Violations',,Violent Crime
  'Harassment',Violent Crime
* 'Counterfeiting',no Violent Crime
* 'Larceny From Motor Vehicle',Violent Crime
* 'Police Service Incidents',Violent Crime
* 'Investigate Person',Violent Crime
* 'Recovered Stolen Property',Violent Crime
* 'Embezzlement',no Violent Crime
* 'Motor Vehicle Accident Response',Violent Crime
* 'Simple Assault',Violent Crime
* 'Warrant Arrests',Violent Crime
* 'Bomb Hoax',Violent Crime
* 'Vandalism',Violent Crime
* 'Missing Person Reported', Violent Crime
* 'License Plate Related Incidents',no Violent Crime
* 'Aggravated Assault',Violent Crime
* 'Property Found',no Violent Crime
* 'Robbery',Violent Crime
* 'Restraining Order Violations',Violent Crime
* 'Property Related Damage',Violent Crime
* 'Missing Person Located',Violent Crime
* 'Landlord/Tenant Disputes',Violent Crime
* 'Disorderly Conduct',Violent Crime
* 'Auto Theft Recovery', no Violent Crime
* 'Medical Assistance', no Violent Crime
* 'License Violation', no Violent Crime
* 'Towed', no Violent Crime
* 'Service',no Violent Crime
* 'Verbal Disputes',Violent Crime
* 'Liquor Violation', no Violent Crime
* 'Fire Related Reports', no Violent Crime
  'Ballistics', Violent Crime
* 'Evading Fare',No Violent Crime
* 'Operating Under the Influence',Violent Crime
* 'Offenses Against Child / Family',Violent Crime
* 'Drug Violation',Violent Crime
* 'Other Burglary',No Violent Crime
* 'Firearm Violations',No Violent Crime
* 'Commercial Burglary', No Violent Crime
* 'Search Warrants', No Violent Crime
* 'Biological Threat', - No Violent Crime
* 'Harbor Related Incidents',- No Violent Crime
* 'Firearm Discovery',- Violent Crime
* 'Prisoner Related Incidents',- Violent Crime
* 'Homicide',- Violent Crime
* 'Assembly or Gathering Violations', - Violent Crime
* 'Manslaughter', - Violent Crime
* 'Arson', - Violent Crime
* 'Criminal Harassment',- Violent Crime
* 'Prostitution',- Violent Crime
* 'HOME INVASION',- Violent Crime
* 'Phone Call Complaints',- No violent crime
* 'Aircraft', - No violent crime
* 'Gambling', - No violent crime
* 'INVESTIGATE PERSON', - No violent crime
* 'Explosives', - Violent Crime
* 'HUMAN TRAFFICKING', - Violent Crime
* 'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE', - Violent Crime
* 'Burglary - No Property Taken' - No violent crime

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

"""
.. module:: src.utils
   :synopsis: TBD

.. moduleauthor:: Jose Angel Velasco - (C) Capgemini Engineering - 2021
"""

import numpy as np


def function(x, y):
    """ *function*
    real function of two variables x and y
    :arg x: first variable
    :arg y: second variable
    :return: value of the function (-1 if error is present)
    """

    try:
        # Check that input parameters are arrays
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)

        # Calculate result
        z = np.round(np.sin(np.sqrt(x ** 2 + y ** 2)), 4)
        return z

    except Exception as exception_msg:
        print('(!) Error in function: ' + str(exception_msg))
        z = -1
        return z


def rosembrock(x, y, a, b):
    """ *rosembrock*
    real function of two variables x and y Rosenbrock
    :arg x: first variable
    :arg y: second variable
    :arg a: second variable
    :arg b: second variable
    :return: value of the function
    """

    try:
        # Check that input parameters are arrays
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        # Calculate result
        z = np.array(np.round((a - x) ** 2 + b * (y - x ** 2) ** 2))
    except Exception as exception_msg:
        print('(!) Error in rosembrock: ' + str(exception_msg))
        z = -1
    return z

"""
.. module:: src.main
   :synopsis: TBD

.. moduleauthor:: Jose Angel Velasco - (C) Capgemini Engineering - 2021
"""
import numpy as np
from src.utils import rosembrock


def main():
    """
        *main*

        This function execute all the code

    """
    x = np.random.normal(1, 0.5, 100)
    y = np.random.normal(5, 0.8, 100)
    a = 1
    b = 100
    f = rosembrock(x, y, a, b)


if __name__ == '__main__':
    main()

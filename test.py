#!/usr/bin/python
# -*- coding: utf-8 -*-
import unittest
import numpy

from module.main import function


class TestFunction(unittest.TestCase):

    def test_output(self):
        """
        Test that the function provide the correct result
        """

        # define input data
        x = numpy.array([4.0237, -0.3734, 1.7, 2.3805, 2.8711, 0.9578, -0.2407,
                         -0.4611, 0.8525, 0.8024])
        y = numpy.array([4.0874, 0.0496, 2.0867, 2.8717, 2.5937, 4.8366, -0.6704,
                         3.3517, 1.8155, 0.4188])
        # get result
        z = function(x=x, y=y)
        z_true = numpy.array([-0.5206, 0.3678, 0.435, -0.5551, -0.6651, -0.9763, 0.6536,
                              -0.2393, 0.9069, 0.7865])
        # Check that the results obtained is equal to expected
        self.assertTrue((z == z_true).all())

    def test_list(self):
        """
        Test that it can work with a list
        """
        # define input data as lists
        x = [4.0237, -0.3734, 1.7, 2.3805, 2.8711, 0.9578, -0.2407,
             -0.4611, 0.8525, 0.8024]
        y = [4.0874, 0.0496, 2.0867, 2.8717, 2.5937, 4.8366, -0.6704,
             3.3517, 1.8155, 0.4188]
        # get result
        z = function(x, y)
        z_true = [-0.5206, 0.3678, 0.435, -0.5551, -0.6651, -0.9763, 0.6536,
                  -0.2393, 0.9069, 0.7865]
        # Check that the results obtained is equal to expected
        self.assertTrue((z == z_true).all())


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/python
# -*- coding: utf-8 -*-
import unittest
from fractions import Fraction

from module.main import function


class TestSum(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        # define input data
        data = [1, 2, 3]
        # get result
        result = function(data)
        # Check that the results obtained is equal to expected
        self.assertEqual(result, 6)

    def test_list_fraction(self):
        """
        Test that it can sum a list of fractions
        """
        # define input data
        data = [Fraction(1, 4), Fraction(1, 4), Fraction(2, 5)]
        # get result
        result = function(data)
        # Check that the results obtained is equal to expected
        self.assertEqual(result, 1)

    def test_bad_type(self):
        data = "banana"
        with self.assertRaises(TypeError):
            result = sum(data)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module:: Tests
   :synopsis: This script define some unitary tests

.. moduleauthor:: Jose Angel Velasco - (C) Tessella Spain by Capgemini Engineering - 2021


"""
import unittest
import pandas
import numpy
import sys
import os


# Move one level up directory
# os.chdir(os.path.dirname(os.getcwd()))


import dummy_project_utils


class TestFunction(unittest.TestCase):
    """
        **Test Case**

        This clase defines several unitary tests to be produced
    """

    def test_load_data(self):
        """
            ** Test Load Data**

            Test that the load data is a data frame

            :param self:
        """

        data_path = 'data\\crimes_dataset.csv'
        logs_path = 'reports\\log_file_data_analyis'
        logger = dummy_project_utils.set_up_logger(logs_path)
        # get result
        df = dummy_project_utils.load_data(data_path, logger)

        # Check that the input loaded is a dataframe
        self.assertTrue(isinstance(df, pandas.DataFrame))

    def test_load_data_empty(self):
        """
            ** Test no-empty data frame**

            Test that the load data is a data frame and is not empty

            :param self:
        """

        # define input data
        data_path = 'data\\crimes_dataset.csv'
        logs_path = 'reports\\log_file_data_analyis'
        logger = dummy_project_utils.set_up_logger(path=logs_path)
        # get result
        df = dummy_project_utils.load_data(path=data_path, logger=logger)

        # Check that the resulting dataframe is not empty
        self.assertFalse(df.empty)


    def test_load_data_empty(self):
        """
            ** Test no-empty data frame**

            Test that the load data is a data frame and is not empty

            :param self:
        """

        # define input data
        data_path = 'data\\crimes_dataset.csv'
        logs_path = 'reports\\log_file_data_analyis'
        logger = dummy_project_utils.set_up_logger(path=logs_path)
        # get result
        df = dummy_project_utils.load_data(path=data_path, logger=logger)

        # Check that the resulting dataframe is not empty
        self.assertFalse(df.empty)


if __name__ == '__main__':
    unittest.main()

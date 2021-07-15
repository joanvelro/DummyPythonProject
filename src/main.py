#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module::Main
   :synopsis: This script ...

.. moduleauthor:: Jose Angel Velasco - (C) Tessella Spain by Capgemini Engineering - 2021


"""
import src.utils.utils

def main():

    logger = src.utils.utils.set_up_logger(path=logs_path + log_file_name)
    logger.info('::: Start Build features :::')


if __name__ == "__main__":
    main()

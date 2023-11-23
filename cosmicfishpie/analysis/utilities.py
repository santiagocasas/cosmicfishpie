# ----------------------------------------------------------------------------------------
#
# This file is part of CosmicFish.
#
# Copyright (C) 2015-2017 by the CosmicFish authors
#
# The CosmicFish code is free software;
# You can use it, redistribute it, and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation;
# either version 3 of the License, or (at your option) any later version.
# The full text of the license can be found in the file LICENSE at
# the top level of the CosmicFish distribution.
#
# ----------------------------------------------------------------------------------------

"""
.. module:: utilities
   :platform: Unix
   :synopsis: A collection of small utilities.

.. moduleauthor:: Marco Raveri <mraveri@uchicago.edu> for the CosmicFish code.
"""

# ***************************************************************************************

import itertools as it
import math
import os

import numpy as np
from scipy.stats.distributions import chi2

# ***************************************************************************************


def num_to_mant_exp(num):
    """
    This function returns the (base 10) exponent and mantissa of a number.

    :param num: input number.
    :type num: :class:`int` or :class:`float`
    :return: tuple (mantissa, exponent) of :class:`int` containing the mantissa and the exponent of the input number.
    :rtype: tuple

    """
    try:
        exponent = math.floor(math.log10(abs(num)))
    except ValueError:  # Case of log10(0)
        return (0, 0)   # Convention: 0 = 0*10^0
    mantissa = num / 10**exponent

    return (mantissa, int(exponent))

# ***************************************************************************************


def mant_exp_to_num(mant_exp):
    """
    This function returns a float built with the given (base 10) mantissa and exponent.

    :param mant_exp: (mantissa, exponent) a tuple of two :class:`int` with the mantissa and the exponent of the input number.
    :type mant_exp: tuple
    :return: output number built as mantissa*10**exponent.
    :rtype: :class:`float`

    """
    return mant_exp[0] * 10**mant_exp[1]

# ***************************************************************************************


def nice_number(num, mode=1, digits=1):
    """
    This function returns a nice number built with num. This is useful to build
    the axes of a plot.
    The nice number is built by taking the first digit of the number.

    :param num: input number
    :type num: :class:`float` or :class:`int`
    :param mode: (optional) operation to use to build the nice number

            | 0 -- use ceil
            | 1 -- use round
            | 2 -- use floor

    :type mode: :class:`int`
    :param digits: input number of digits to keep
    :type digits: :class:`int`
    :return: a nice number!
    :rtype: :class:`float`

    """
    # extract mantissa and exponent:
    mant, exp = num_to_mant_exp(num)
    # select the working mode and do the truncation:
    if (mode == 0):
        mant = np.ceil(mant * 10**(digits - 1)) / 10**(digits - 1)
    elif (mode == 1):
        mant = np.round(mant, digits - 1)
    elif (mode == 2):
        mant = np.floor(mant * 10**(digits - 1)) / 10**(digits - 1)
    else:
        raise ValueError('Wrong worging mode for Fisher_utilities.nice_number')

    return mant_exp_to_num((mant, exp))


# def nice_number( num, mode=0 ):
#    """
#    This function returns a nice number built with num. This is useful to build the axes of a plot.
#    The nice number is built by taking the first digit of the number.
#
#    :param num: input number
#    :type num: :class:`float` or :class:`int`
#    :param mode: (optional) operation to use to build the nice number
#
#            | 0 -- use ceil
#            | 1 -- use round
#            | 2 -- use floor
#
#    :type mode: :class:`int`
#    :return: a nice number!
#    :rtype: :class:`float`
#
#    """
#    # extract the mantissa
#    exponent = num_to_mant_exp( num )[1]
#    # select the working mode
#    if ( mode==0 ):
#        mantissa = np.ceil( num_to_mant_exp( num )[0])
#    elif ( mode==1 ):
#        mantissa = np.round( num_to_mant_exp( num )[0])
#    elif ( mode==2 ):
#        mantissa = np.floor( num_to_mant_exp( num )[0])
#    else:
#        raise ValueError( 'Wrong worging mode for Fisher_utilities.nice_number' )
#
#    return mant_exp_to_num( ( mantissa, exponent ) )

v_nice_number = np.vectorize(nice_number)

# ***************************************************************************************


def significant_digits(num_err, mode=1, digits=1):
    """
    This function returns the number in num_err at the precision of error.

    :param num_err: (number, error) input number and error in a tuple.
    :type num_err: tuple
    :param mode: (optional) operation to use to build the number

            | 0 -- use ceil
            | 1 -- use round
            | 2 -- use floor

    :type mode: :class:`int`
    :param digits: input number of digits to keep
    :type digits: :class:`int`
    :return: a number with all the significant digits according to error
    :rtype: :class:`float`

    """
    number = num_err[0]
    error = num_err[1]
    number_mant_exp = num_to_mant_exp(number)
    error_mant_exp = num_to_mant_exp(error)

    temp = mant_exp_to_num(
        (number_mant_exp[0],
         number_mant_exp[1] -
         error_mant_exp[1]))
    # select the working mode
    if (mode == 0):
        temp = np.ceil(temp * 10**(digits - 1)) / 10**(digits - 1)
    elif (mode == 1):
        temp = np.round(temp, digits - 1)
    elif (mode == 2):
        temp = np.floor(temp * 10**(digits - 1)) / 10**(digits - 1)
    else:
        raise ValueError(
            'Fisher_utilities.significant_digits called with mode=' +
            str(mode) +
            ' legal values are 0,1,2')

    return temp * 10**(error_mant_exp[1])
# ***************************************************************************************


def confidence_coefficient(confidence_level, dimensions=1):
    """
    This function returns the number of sigmas given a confidence level.
     See page 815 of Numerical Recipes, Press et al., 2007
     Uses the inverse CDF of the chi squared distribution
    :param confidence_level: desired confidence level. Between 0 and 1.
    :type confidence_level: :class:`float`
    :return: the coefficient (number of sigmas) for the desired confidence level.
    :rtype: :class:`float`

    """
    return np.sqrt(chi2.ppf(confidence_level, df=dimensions))


# ***************************************************************************************

def print_table(table):
    """
    This function prints on the screen a nicely formatted table.

    :param table: a 2D list that should be printed on the screen.

    """
    # transpose the table:
    table = list(map(list, list(zip(*table))))
    # get the column width:
    col_width = [max(len(str(x)) for x in col) for col in zip(*table)]
    # print it to screen:
    print()
    for line in table:
        print(
            "| " +
            " | ".join(
                "{:{}}".format(
                    x,
                    col_width[i]) for i,
                x in enumerate(line)) +
            " |")
    print()

# ***************************************************************************************


def make_list(elements):
    """
    Checks if elements is a list.
    If yes returns elements without modifying it.
    If not creates and return a list with elements inside.

    :param elements: an element or a list of elements
    :return: a list containing elements if elements is not a list, elements otherwise.
    :rtype: list

    """
    if isinstance(elements, (list, tuple)):
        return elements
    else:
        return [elements]

# ***************************************************************************************


def grouper(n, iterable, fillvalue=None):
    """
    This small function regroups a list in sub lists of n elements

    :param n: an element or a list of elements
    :param iterable: input list
    :param fillvalue: value to put to fill if no element is present
    :return: a list of list containing grouped elements
    :rtype: list

    """
    args = [iter(iterable)] * n
    return list(it.izip_longest(fillvalue=fillvalue, *args))

# ***************************************************************************************


def find_nearest(array, value):
    """
    This function finds the index of the element in array which is nearest to ValueError
    :param array:  an array of numbers
    :param value:  a number
    :return: the index in array nearest to value
    :rtype:   int
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def mkdirp(dirpath):
    """
    This function creates the directory dirpath if it is not found
    :param dirpath: string with the path of the directory to be created
    :return:  None
    :rtype:   NoneType
    """
    outdir = os.path.dirname(dirpath)
    # create directory if it does not exist
    if outdir == '':
        print('Output root is on working directory')
    elif not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
            print((str(outdir) + ' directory created'))
        except OSError :
                raise
    else:
        print((str(outdir) + '  exists already'))
    return None


def rel_median_error(array, percentage=True):
    """
    This function returns the percentage difference compared to median of an array for each element
    :param array:  numpy array of numbers
    :optional_param percentage: if set to True, returns percentage difference. Default: True
    :return: difference of each element compared to median of array
    :rtype: Numpy array
    """
    perfact = 1
    if percentage:
        perfact = 100
    median = np.median(np.array(array), 0)
    relerr = perfact * (array - median) / median
    return relerr


def CosmicFish_write_header(name):
    """
    This function prints to screen the CosmicFish header.
    To be called at the beginning of the applications.

    :param name: string that contains the name of the program. This will be printed
        along the CosmicFish header.

    """

    print()
    print("**************************************************************")
    print("   _____               _     _____     __  ")
    print("  / ___/__  ___ __ _  (_)___/ __(_)__ / /  ")
    print(" / /__/ _ \\(_-</  ' \\/ / __/ _// (_-</ _ \\ ")
    print(" \\___/\\___/___/_/_/_/_/\\__/_/ /_/___/_//_/ Py Lib")
    print(" ")
    print("**************************************************************")
    print(name)
    print(" This application was developed using the CosmicFish code.")
    print("**************************************************************")
    print()

# ***************************************************************************************

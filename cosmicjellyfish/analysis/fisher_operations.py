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
:synopsis: Module that contains operations that can be performed on Fisher matrices.
           All of them are safeguarded against non-Fisher input.
"""

# ***************************************************************************************

import math

import numpy as np

from cosmicjellyfish.analysis import fisher_matrix as fm

# ***************************************************************************************


def eliminate_columns_rows(fisher_matrix, indexes):
    """
    This function eliminates the row and columns corresponding to the given indexes
    from the Fisher matrix. It also deletes all the other informations like the names
    of the parameters. Notice that the index corresponding to the first parameter
    is zero.

    :param fisher_matrix: input Fisher matrix
    :type fisher_matrix: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`
    :param indexes: list of integers with the indexes to delete from the Fisher matrix
    :type indexes: :class:`list` of :class:`int`
    :returns: A Fisher matrix with the columns and rows deleted
    :rtype: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`

    """
    # check validity of the input:
    if not isinstance(fisher_matrix, fm.fisher_matrix):
        raise ValueError("Error, input fisher_matrix is not a fisher_matrix")
    # write the param names:
    new_param_names = []
    new_param_names_latex = []
    new_param_fiducial = []
    for i in range(fisher_matrix.num_params):
        if i not in indexes:
            new_param_names.append(fisher_matrix.param_names[i])
            new_param_names_latex.append(fisher_matrix.param_names_latex[i])
            new_param_fiducial.append(fisher_matrix.param_fiducial[i])
    # write the Fisher matrix:
    fisher_temp = np.delete(np.delete(fisher_matrix.fisher_matrix, indexes, 0), indexes, 1)
    # initialize the new Fisher matrix:
    fisher_new = fm.fisher_matrix(
        fisher_matrix=fisher_temp,
        param_names=new_param_names,
        param_names_latex=new_param_names_latex,
        fiducial=new_param_fiducial,
    )
    fisher_new.name = fisher_matrix.name + "_reduced"
    fisher_new.path = fisher_matrix.path
    fisher_new.indir = fisher_matrix.indir

    return fisher_new


# ***************************************************************************************


def eliminate_parameters(fisher_matrix, names):
    """
    This function eliminates the row and columns corresponding to the given parameter
    name from the Fisher matrix. It also deletes all the other informations like the names
    of the parameters.

    :param fisher_matrix: input Fisher matrix
    :type fisher_matrix: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`
    :param names: list of names of the parameters to delete from the Fisher matrix
    :type names: :class:`list` of :class:`string`
    :returns: A Fisher matrix with the parameters deleted
    :rtype: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`

    """
    # check validity of the input:
    if not isinstance(fisher_matrix, fm.fisher_matrix):
        raise ValueError("Error, input fisher_matrix is not a fisher_matrix")
    # get the indexes of the parameters:
    index_list = []
    for i in names:
        if i not in fisher_matrix.param_names_dict:
            raise ValueError(
                "Error, parameter " + str(i) + " is not in a parameter of fisher_matrix"
            )
        index_list.append(fisher_matrix.param_names_dict[i] - 1)
    # elminate them from the list and return:
    return eliminate_columns_rows(fisher_matrix, index_list)


# ***************************************************************************************


def reshuffle(fisher_matrix, names, update_names=True):
    """
    This function reshuffles a Fisher matrix. The new Fisher matrix will have the
    parameters specified in names, in the order specified by names.
    Can be used to delete parameters, change their order or extract the Fisher
    for some parameters without marginalizing over the others.

    :param fisher_matrix: input Fisher matrix
    :type fisher_matrix: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`
    :param names: list of names of the parameters that are desired in the output Fisher
        matrix, in the desired order.
    :type names: :class:`list` of :class:`string`
    :returns: A Fisher matrix with the new parameters
    :rtype: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`

    """
    # check validity of the input:
    if not isinstance(fisher_matrix, fm.fisher_matrix):
        raise ValueError("Error, input fisher_matrix is not a fisher_matrix")
    # check wether the names required are inside the Fisher matrix:
    for i in names:
        if i not in fisher_matrix.param_names_dict:
            raise ValueError(
                "Error, parameter " + str(i) + " is not in a parameter of fisher_matrix"
            )
    # get the new latex names and fiducial:
    new_param_names_latex = []
    new_param_fiducial = []
    for i in names:
        ind = fisher_matrix.param_names_dict[i] - 1
        new_param_names_latex.append(fisher_matrix.param_names_latex[ind])
        new_param_fiducial.append(fisher_matrix.param_fiducial[ind])
    # initialize an empty matrix:
    num_param_new = len(names)
    new_matrix = np.zeros([num_param_new, num_param_new])
    # fill the new matrix:
    for i in range(num_param_new):
        for j in range(num_param_new):
            # get the name:
            x = names[i]
            y = names[j]
            # get the parameter name:
            x1 = fisher_matrix.param_names_dict[x] - 1
            y1 = fisher_matrix.param_names_dict[y] - 1
            # get the entrance of the new matrix:
            new_matrix[i, j] = fisher_matrix.fisher_matrix[x1, y1]
    # create the new Fisher matrix:
    fisher_new = fm.fisher_matrix(
        fisher_matrix=new_matrix,
        param_names=names,
        param_names_latex=new_param_names_latex,
        fiducial=new_param_fiducial,
    )
    if update_names:
        fisher_new.name = fisher_matrix.name + "_reshuffled"
    else:
        fisher_new.name = fisher_matrix.name

    fisher_new.path = fisher_matrix.path
    fisher_new.indir = fisher_matrix.indir

    return fisher_new


# ***************************************************************************************


def marginalise(fisher_matrix, names, update_names=True):
    """
    This function marginalises a Fisher matrix over all parameters but the ones in names.
    The new Fisher matrix will have the parameters specified in names, in the order specified by names.
    The calculation is performed in the numerically stable way.

    :param fisher_matrix: input Fisher matrix
    :type fisher_matrix: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`
    :param names: list of names of the parameters of the output Fisher matrix,
        in the order that will appear in the output Fisher matrix. All other parameters
        will be marginalized over.
    :type names: :class:`list` of :class:`string`
    :returns: A Fisher matrix with the marginalized parameters
    :rtype: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`

    """
    # check validity of the input:
    if not isinstance(fisher_matrix, fm.fisher_matrix):
        raise ValueError("Error, input fisher_matrix is not a fisher_matrix")
    # check wether the names required are inside the Fisher matrix:
    for i in names:
        if i not in fisher_matrix.param_names_dict:
            raise ValueError(
                "Error, parameter " + str(i) + " is not in a parameter of fisher_matrix"
            )
    # get the new latex names and fiducial:
    new_param_names_latex = []
    new_param_fiducial = []
    for i in names:
        ind = fisher_matrix.param_names_dict[i] - 1
        new_param_names_latex.append(fisher_matrix.param_names_latex[ind])
        new_param_fiducial.append(fisher_matrix.param_fiducial[ind])
    # initialize an empty matrix:
    num_param_new = len(names)
    new_matrix = np.zeros([num_param_new, num_param_new])
    # fill the new inverse matrix:
    for i in range(num_param_new):
        for j in range(num_param_new):
            # get the name:
            x = names[i]
            y = names[j]
            # get the parameter name:
            x1 = fisher_matrix.param_names_dict[x] - 1
            y1 = fisher_matrix.param_names_dict[y] - 1
            # get the entrance of the new matrix:
            new_matrix[i, j] = fisher_matrix.get_fisher_inverse()[x1, y1]

    fisher_temp = np.linalg.inv(new_matrix)
    # create the new Fisher matrix:
    fisher_new = fm.fisher_matrix(
        fisher_matrix=fisher_temp,
        param_names=names,
        param_names_latex=new_param_names_latex,
        fiducial=new_param_fiducial,
        name=fisher_matrix.name,
    )
    if update_names:
        fisher_new.name = fisher_matrix.name + "_marginal"
    fisher_new.path = fisher_matrix.path
    fisher_new.indir = fisher_matrix.indir

    return fisher_new


# ***************************************************************************************


def marginalise_over(fisher_matrix, names):
    """
    This function marginalises a Fisher matrix over the parameters in names.
    The new Fisher matrix will not have the parameters specified in names.
    The calculation is performed in the numerically stable way.

    :param fisher_matrix: input Fisher matrix
    :type fisher_matrix: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`
    :param names: list of names of the parameters over which the Fisher will be marginalised.
    :type names: :class:`list` of :class:`string`
    :returns: A Fisher matrix with the names parameters marginalized.
    :rtype: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`

    """
    # check validity of the input:
    if not isinstance(fisher_matrix, fm.fisher_matrix):
        raise ValueError("Error, input fisher_matrix is not a fisher_matrix")
    # check wether the names required are inside the Fisher matrix:
    for i in names:
        if i not in fisher_matrix.param_names_dict:
            raise ValueError(
                "Error, parameter " + str(i) + " is not in a parameter of fisher_matrix"
            )
    # get the indexes:
    new_names = [i for i in fisher_matrix.param_names if i not in names]

    return marginalise(fisher_matrix, new_names)


# ***************************************************************************************


def information_gain(fisher_1, fisher_2, fisher_prior, units=math.log(2.0), stat=True):
    """
    This function computes the Fisher approximation of Kullback-Leibler information gain.
    For the details of the formula we refer to the CosmicFish notes.

    :param fisher_1: first input Fisher matrix
    :type fisher_1: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`
    :param fisher_2: second input Fisher matrix
    :type fisher_2: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`
    :param fisher_prior: input Fisher matrix with the prior information.
    :type fisher_prior: :class:`cosmicfish_pylib.fisher_matrix.fisher_matrix`
    :param units: Units of information gain. Optional by default in Bits.
    :type units: :class:`float`
    :param stat: wether to output the expected value and variance
    :type stat: :class:`logical`
    :returns: a :class:`float` with the information gain.
    :rtype: :class:`float`
    """
    info_gain = 0.0
    # first computations:
    F1p = fisher_1 + fisher_prior
    F2p = fisher_2 + fisher_prior
    # get common parameter names:
    param_names = [name for name in F1p.get_param_names() if name in F2p.get_param_names()]
    # reshuffle the second matrix:
    F1p = reshuffle(F1p, param_names)
    F2p = reshuffle(F2p, param_names)
    # define a dummy Fisher matrix with empty entrances and with the same
    # parameters as the others:
    fisher_temp = fm.fisher_matrix(
        fisher_matrix=0.0 * F2p.get_fisher_matrix(),
        param_names=F2p.get_param_names(),
        param_names_latex=F2p.get_param_names_latex(),
        fiducial=F2p.get_param_fiducial(),
    )
    fisher_temp = fisher_2 + fisher_temp
    # the first term:
    info_gain = info_gain - math.log(F1p.determinant() / F2p.determinant())
    info_gain = info_gain - F1p.get_fisher_matrix().shape[0]
    # the second trace term:
    info_gain = info_gain + np.trace(np.dot(F2p.get_fisher_inverse(), F1p.get_fisher_matrix()))
    # add additional term if statistical average over data is wanted
    if stat:
        # we break down the third term into two pieces:
        temp = np.dot(
            np.dot(
                np.dot(fisher_temp.get_fisher_matrix(), F2p.get_fisher_inverse()),
                F1p.get_fisher_matrix(),
            ),
            F2p.get_fisher_inverse(),
        )
        temp = temp + np.dot(
            np.dot(temp, fisher_temp.get_fisher_matrix()), F1p.get_fisher_inverse()
        )
        info_gain = info_gain + np.trace(temp)
        # compute variance:
        temp = np.dot(temp, temp)
    # output
    info_gain = info_gain / 2.0 / units
    return info_gain


# ***************************************************************************************


# ***************************************************************************************


def jacobian_from_function(function, fiducial, method="fd", step=1.0e-6):
    r"""
    Jacobian of a vector function at a fiducial point.

    The returned matrix is :math:`J_{i\alpha} = \partial f_i / \partial x_\alpha`, with rows
    indexing the outputs of ``function`` and columns indexing its inputs.

    :param function: callable mapping a 1D array of inputs to a 1D array of outputs.
        With ``method='jax'`` it must be written with ``jax.numpy``.
    :type function: callable
    :param fiducial: point where the Jacobian is evaluated.
    :type fiducial: array-like
    :param method: ``'fd'`` for central finite differences or ``'jax'`` for forward-mode
        automatic differentiation (requires jax).
    :type method: :class:`string`
    :param step: relative finite-difference step; the absolute step for input
        :math:`x_\alpha` is ``step * max(|x_alpha|, 1)``. Ignored with ``method='jax'``.
    :type step: :class:`float`
    :returns: the Jacobian matrix, shape ``(n_outputs, n_inputs)``.
    :rtype: :class:`numpy.ndarray`
    """
    point = np.asarray(fiducial, dtype=float)
    if point.ndim != 1:
        raise ValueError("Error, fiducial must be a 1D array")
    if method == "jax":
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "method='jax' requires jax. Install it with: pip install 'cosmicjellyfish[jaxcosmo]'"
            ) from err
        return np.atleast_2d(np.asarray(jax.jacfwd(function)(jnp.asarray(point))))
    if method != "fd":
        raise ValueError("Error, method must be 'fd' or 'jax'")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Error, finite-difference step must be finite and positive")
    columns = []
    for index, value in enumerate(point):
        delta = step * max(abs(value), 1.0)
        plus = point.copy()
        minus = point.copy()
        plus[index] += delta
        minus[index] -= delta
        diff = np.asarray(function(plus), dtype=float) - np.asarray(function(minus), dtype=float)
        columns.append(np.atleast_1d(diff) / (2.0 * delta))
    return np.stack(columns, axis=1)


# ***************************************************************************************


def reparametrise(
    fisher_matrix, jacobian, new_names, new_names_latex=None, new_fiducial=None, name=None
):
    r"""
    Change the parameters of a Fisher matrix with the chain rule.

    For old parameters :math:`p_i` that are functions of new parameters :math:`q_\alpha`,
    the Fisher matrix in the new parameters is

    .. math::

        \tilde F_{\alpha\beta} = \sum_{ij} J_{i\alpha} F_{ij} J_{j\beta},
        \qquad J_{i\alpha} = \frac{\partial p_i}{\partial q_\alpha}.

    This is the projection used, for example, to go from the redshift-binned
    :math:`H(z_i)`, :math:`D_A(z_i)`, :math:`f\sigma_8(z_i)` constraints of a galaxy
    clustering forecast to cosmological parameters. The number of new parameters can be
    smaller than the number of old ones.

    This is different from :class:`cosmicjellyfish.analysis.fisher_derived.fisher_derived`,
    which propagates the *covariance* to derived parameters :math:`d_a(p)` through
    :math:`A^T F^{-1} A` with :math:`A_{ia} = \partial d_a / \partial p_i`. Use this function
    when the old parameters are functions of the new ones; use ``fisher_derived`` when the
    new parameters are functions of the old ones.

    :param fisher_matrix: input Fisher matrix in the old parameters.
    :type fisher_matrix: :class:`cosmicjellyfish.analysis.fisher_matrix.fisher_matrix`
    :param jacobian: matrix :math:`\partial p_i / \partial q_\alpha` with shape
        ``(n_old, n_new)``; rows follow ``fisher_matrix.param_names``.
    :type jacobian: :class:`numpy.ndarray`
    :param new_names: names of the new parameters, one per Jacobian column.
    :type new_names: :class:`list` of :class:`string`
    :param new_names_latex: optional LaTeX names of the new parameters.
    :type new_names_latex: :class:`list` of :class:`string`
    :param new_fiducial: optional fiducial values of the new parameters.
    :type new_fiducial: :class:`list` of :class:`float`
    :param name: optional name of the output Fisher matrix; defaults to the input name
        with ``_reparametrised`` appended.
    :type name: :class:`string`
    :returns: the Fisher matrix in the new parameters.
    :rtype: :class:`cosmicjellyfish.analysis.fisher_matrix.fisher_matrix`
    """
    if not isinstance(fisher_matrix, fm.fisher_matrix):
        raise ValueError("Error, input fisher_matrix is not a fisher_matrix")
    jac = np.asarray(jacobian, dtype=float)
    new_names = list(new_names)
    if jac.ndim != 2 or jac.shape[0] != fisher_matrix.num_params:
        raise ValueError(
            "Error, jacobian must have shape (n_old, n_new) with n_old = "
            + str(fisher_matrix.num_params)
        )
    if jac.shape[1] != len(new_names):
        raise ValueError("Error, the number of new names must match the jacobian columns")
    duplicates = sorted({name for name in new_names if new_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Error, new parameter names must be unique; duplicates: {duplicates}")
    new_matrix = jac.T @ fisher_matrix.get_fisher_matrix() @ jac
    new_matrix = 0.5 * (new_matrix + new_matrix.T)
    fisher_new = fm.fisher_matrix(
        fisher_matrix=new_matrix,
        param_names=new_names,
        param_names_latex=None if new_names_latex is None else list(new_names_latex),
        fiducial=None if new_fiducial is None else list(new_fiducial),
        name=fisher_matrix.name + "_reparametrised" if name is None else name,
    )
    fisher_new.path = fisher_matrix.path
    fisher_new.indir = fisher_matrix.indir
    return fisher_new

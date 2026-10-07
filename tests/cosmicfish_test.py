import numpy as np

from cosmicfishpie.analysis.fisher_operations import marginalise_over
from cosmicfishpie.utilities.utils import printing as cpr


def test_FisherMatrix_GCsp(spectro_fisher_matrix):
    cpr.debug = False
    fish = spectro_fisher_matrix.compute(max_z_bins=1)
    assert set(spectro_fisher_matrix.freeparams) <= set(spectro_fisher_matrix.derivs_dict)
    print("Fisher name: ", fish.name)
    print("Fisher parameters: ", fish.get_param_names())
    print("Fisher fiducial values: ", fish.get_param_fiducial())
    print("Fisher confidence bounds: ", fish.get_confidence_bounds())
    print("Fisher covariance matrix: ", fish.fisher_matrix_inv)
    assert hasattr(fish, "name")
    assert hasattr(fish, "fisher_matrix")
    assert hasattr(fish, "fisher_matrix_inv")
    assert hasattr(fish, "get_confidence_bounds")
    assert np.isclose(fish.fisher_matrix[0, 0], 218868.67901548574, rtol=1e-3)
    assert np.isclose(fish.fisher_matrix[3, 3], 0.00834065097, rtol=1e-3)
    assert np.isclose(fish.fisher_matrix[1, 3], 12.3776463, rtol=1e-3)
    assert np.isclose(fish.fisher_matrix[3, 1], 12.3776463, rtol=1e-3)
    assert np.isclose(fish.fisher_matrix[2, 1], 54494.1330, rtol=1e-3)
    assert np.isclose(np.sqrt(fish.fisher_matrix_inv[0, 0]), 0.00798381010, rtol=1e-3)


def test_marginalise_over(spectro_fisher_matrix):
    fish = spectro_fisher_matrix.compute(max_z_bins=1)
    marginalized_fish = marginalise_over(fish, ["h"])
    assert hasattr(marginalized_fish, "fisher_matrix")
    assert hasattr(marginalized_fish, "fisher_matrix_inv")
    assert hasattr(marginalized_fish, "get_confidence_bounds")
    assert hasattr(marginalized_fish, "get_param_names")
    assert len(marginalized_fish.get_param_names()) == (1 + 2)

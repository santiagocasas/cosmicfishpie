import numpy as np

from cosmicfishpie.analysis.fisher_operations import marginalise_over
from cosmicfishpie.utilities.utils import printing as cpr


def test_FisherMatrix_GCsp(spectro_fisher_matrix):
    cpr.debug = False
    fish = spectro_fisher_matrix.compute(max_z_bins=1)
    print("Fisher name: ", fish.name)
    print("Fisher parameters: ", fish.get_param_names())
    print("Fisher fiducial values: ", fish.get_param_fiducial())
    print("Fisher confidence bounds: ", fish.get_confidence_bounds())
    print("Fisher covariance matrix: ", fish.fisher_matrix_inv)
    assert hasattr(fish, "name")
    assert hasattr(fish, "fisher_matrix")
    assert hasattr(fish, "fisher_matrix_inv")
    assert hasattr(fish, "get_confidence_bounds")
    assert np.isclose(fish.fisher_matrix[0, 0], 216935.5, rtol=1e-3)
    assert np.isclose(fish.fisher_matrix[3, 3], 0.0083476, rtol=1e-3)
    assert np.isclose(fish.fisher_matrix[1, 3], 12.4387, rtol=1e-3)
    assert np.isclose(fish.fisher_matrix[3, 1], 12.4387, rtol=1e-3)
    assert np.isclose(fish.fisher_matrix[2, 1], 55348.06, rtol=1e-3)
    assert np.isclose(np.sqrt(fish.fisher_matrix_inv[0, 0]), 0.00793561215, rtol=1e-3)


def test_marginalise_over(spectro_fisher_matrix):
    fish = spectro_fisher_matrix.compute(max_z_bins=1)
    marginalized_fish = marginalise_over(fish, ["h"])
    assert hasattr(marginalized_fish, "fisher_matrix")
    assert hasattr(marginalized_fish, "fisher_matrix_inv")
    assert hasattr(marginalized_fish, "get_confidence_bounds")
    assert hasattr(marginalized_fish, "get_param_names")
    assert len(marginalized_fish.get_param_names()) == (1 + 2)

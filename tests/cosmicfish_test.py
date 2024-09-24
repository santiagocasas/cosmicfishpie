import numpy as np

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
    assert np.isclose(fish.fisher_matrix[0, 0], 216309.21975347662)
    assert np.isclose(fish.fisher_matrix[3, 3], 0.008400962234855812)
    assert np.isclose(fish.fisher_matrix[1, 3], -7.045002349409897)
    assert np.isclose(fish.fisher_matrix[3, 1], -7.045002349409897)
    assert np.isclose(np.sqrt(fish.fisher_matrix_inv[0, 0]), 0.005893992229989306)

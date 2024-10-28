import numpy as np

from cosmicfishpie.utilities.utils import printing as cpr


def test_FisherMatrix_WL(photo_fisher_matrix):
    cpr.debug = False
    fish = photo_fisher_matrix.compute()
    print("Fisher name: ", fish.name)
    print("Fisher parameters: ", fish.get_param_names())
    print("Fisher fiducial values: ", fish.get_param_fiducial())
    print("Fisher confidence bounds: ", fish.get_confidence_bounds())
    print("Fisher covariance matrix: ", fish.fisher_matrix_inv)
    assert hasattr(fish, "name")
    assert hasattr(fish, "fisher_matrix")
    assert hasattr(fish, "fisher_matrix_inv")
    assert hasattr(fish, "get_confidence_bounds")
    assert np.isclose(fish.fisher_matrix[0, 0], 6604357.453651955)
    assert np.isclose(fish.fisher_matrix[1, 1], 127456.10774308711)
    assert np.isclose(fish.fisher_matrix[3, 8], 26.590294486780888)
    assert np.isclose(fish.fisher_matrix[8, 3], 26.590294486780888)
    assert np.isclose(fish.fisher_matrix[9, 10], 18285.415448376294)
    assert np.isclose(np.sqrt(fish.fisher_matrix_inv[0, 0]), 0.0005291588745873003)

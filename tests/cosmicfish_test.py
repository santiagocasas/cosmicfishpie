from cosmicfishpie.utilities.utils import printing as cpr


def test_FisherMatrix(spectro_fisher_matrix):
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
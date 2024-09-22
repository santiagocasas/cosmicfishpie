from cosmicfishpie.LSSsurvey.photo_obs import ComputeCls

# You might need to add more imports depending on what you're testing


def test_compute_cls_initialization(photo_fisher_matrix):
    cosmopars = {"Omegam": 0.3, "h": 0.7}  # Add more parameters as needed
    cosmoFM = photo_fisher_matrix
    photo_Cls = ComputeCls(cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.biaspars)
    assert isinstance(photo_Cls, ComputeCls)
    assert photo_Cls.cosmopars == cosmopars


def test_getcls(photo_fisher_matrix):
    cosmoFM = photo_fisher_matrix
    cosmopars = {"Omegam": 0.3, "h": 0.7}  # Add more parameters as needed
    photo_Cls = ComputeCls(cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.biaspars)
    photo_Cls.compute_all()
    assert isinstance(photo_Cls.result, dict)
    # Add more specific assertions based on what you expect in the result

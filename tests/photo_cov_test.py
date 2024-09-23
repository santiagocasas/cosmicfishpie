from cosmicfishpie.LSSsurvey.photo_cov import PhotoCov

# You might need to add more imports depending on what you're testing


def test_photo_cov_initialization(photo_fisher_matrix):
    cosmopars = {"Omegam": 0.3, "h": 0.7}  # Add more parameters as needed
    cosmoFM = photo_fisher_matrix
    photo_cov = PhotoCov(cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.biaspars)
    assert isinstance(photo_cov, PhotoCov)
    assert photo_cov.cosmopars == cosmopars


def test_get_cls(photo_fisher_matrix):
    cosmoFM = photo_fisher_matrix
    cosmopars = {"Omegam": 0.3, "h": 0.7}  # Add more parameters as needed
    photo_cov = PhotoCov(cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.biaspars)
    allparsfid = dict()
    allparsfid.update(cosmopars)
    allparsfid.update(cosmoFM.IApars)
    allparsfid.update(cosmoFM.biaspars)
    allparsfid.update(cosmoFM.photopars)
    cls = photo_cov.getcls(allparsfid)
    assert isinstance(cls, dict)
    # Add more specific assertions based on what you expect in the result


def test_get_cls_noise(photo_fisher_matrix):
    cosmoFM = photo_fisher_matrix
    cosmopars = {"Omegam": 0.3, "h": 0.7}  # Add more parameters as needed
    photo_cov = PhotoCov(cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.biaspars)
    cls = photo_cov.getcls(photo_cov.allparsfid)
    noisy_cls = photo_cov.getclsnoise(cls)
    assert isinstance(noisy_cls, dict)
    # Add more specific assertions based on what you expect in the result

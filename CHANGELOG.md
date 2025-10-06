# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
1.0.1 : Bugfix for external file reading for release notebooks.
        Change `is False` statements with a normal `not` to catch False-equivalents.

1.0.2 : Added documentation for the cosmicfishpie.LSSsurvey submodule

1.0.3 : Added documentation for the cosmicfishpie.fishermatrix submodule

1.0.4 : Merged changes from the old repo to the release repo.
        Mainly changes to the plotting routine
        Added new external file for nonlinear power spectra
        New options for spectroscopic error. Removed infamous h-bug

1.0.5 : Removed a backwards compatibility issue with older version of numpy.
        Switched Integration to use scipy now

1.0.6 : Restructured default parameters for EBS and Survey Specifications

1.0.7 : Added a new installation script to download external data from the github
1.0.8 : Overview documentation and initial tests
1.1.0 : Implemented symbolic as a new boltzmann solver class. Fixes of feedback prints. More docstrings.
1.1.1 : Resolved gcsp bias bug, new test suite. 
1.1.2 : Coverage badge
1.1.3 : New split up demonstration notebooks
1.1.4 : Coverage reports with Codecov
1.2.0 : Big update of configuration, specification yamls and nuisance parameter interface. No backwards compatibility of yamls!
1.2.1 : Updating configs of other surveys: SKAO, DESI, LSST to work in new config file structure
1.2.2 : Galaxy sample split for sources and lenses. Feedback prints more consistent.
1.2.3 : Added a new likelihood function for photometric and spectroscopic surveys.
1.2.4 : Discontinued support for python3.8. Fixed the style for likelihood scripts.
1.2.5 : Likelihood modules tested for photo and spectro
1.2.6 : New demo notebook GCsp
1.2.7 : Added test for analysis module
1.2.8 : Added compehensive test catalogue. 

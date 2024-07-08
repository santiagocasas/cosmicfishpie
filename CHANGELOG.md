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

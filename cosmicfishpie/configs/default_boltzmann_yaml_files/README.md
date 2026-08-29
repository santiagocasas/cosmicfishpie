# Structure of the folder

In this folder, the individual sub folders contain some additional arguments passed to the Einstein--Boltzmann solver to obtain the results of the `cosmology` class.
Each sub folder is specifically for one EBS code. The folders `class` and `camb` have, additionally to the default files, also files specifically to reproduce the results in papers:

    - Casas, S et al. [2303.09451]
    - Archidiacono, M et al. [2405.06047]

They are called `mpvalidation.yaml` and `nuvalidation.yaml`, respectively. For the case of `class` the file `nuvalidation.yaml` is split again, as explained in the paper.

## Paper neutrino precision profiles

The canonical precision settings for the neutrino validation are stored centrally in
`precision_profiles.yaml`. It records the provenance from Archidiacono et al.
[2405.06047v1] and defines `camb_hp`, `class_hp`, and `class_uhp`.

Paper solver YAMLs reference a profile with `precision_profile_file` and
`precision_profile`. Their remaining contents contain only model-specific settings.
The loader merges local YAML values over the selected profile, so intentional
experiments can override individual settings explicitly. For example,
`paper_mnuvalidation_photo_HP.yaml` is a stricter sensitivity variant of `class_hp`,
not the paper's canonical photo HP profile.

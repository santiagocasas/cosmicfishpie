# Structure of the folder

In this folder, the individual sub folders contain some additional arguments passed to the Einstein--Boltzmann solver to obtain the results of the `cosmology` class.
Each sub folder is specifically for one EBS code. The folders `class` and `camb` have, additionally to the default files, also files specifically to reproduce the results in papers:

    - Casas, S et al. [2303.09451]
    - Archidiacono, M et al. [2405.06047]

They are called `mpvalidation.yaml` and `nuvalidation.yaml`, respectively. For the case of `class` the file `nuvalidation.yaml` is split again, as explained in the paper.

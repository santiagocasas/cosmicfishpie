import os

import requests

import cosmicfishpie


def _install_euclid_lumratio(installpath):
    remote_url = "https://raw.githubusercontent.com/santiagocasas/cosmicfishpie/main/survey_specifications/lumratio_file.dat"
    install_ext_data(installpath, remote_url)


def _install_ska_data(installpath):
    remote_url = "https://raw.githubusercontent.com/santiagocasas/cosmicfishpie/main/survey_specifications/SKA1-IM-z_Pnoise.txt"
    install_ext_data(installpath, remote_url)
    remote_url = "https://raw.githubusercontent.com/santiagocasas/cosmicfishpie/main/survey_specifications/SKA1_THI_sys_noise.txt"
    install_ext_data(installpath, remote_url)
    remote_url = "https://raw.githubusercontent.com/santiagocasas/cosmicfishpie/main/survey_specifications/nofz_ska1_5000.dat"
    install_ext_data(installpath, remote_url)


def install_ext_data(installpath, remote_url):
    filename = remote_url.split("/")[-1]
    filepath = os.path.join(installpath, filename)
    remote = requests.get(remote_url)
    file = open(filepath, "w")
    file.write(remote.text)


if __name__ == "__main__":
    userinput = input(
        "External data is going to be downloaded to:\n {}.\n Do you want to proceed? (y/N)\n ".format(
            os.path.join(cosmicfishpie.__path__[0], "configs", "external_data")
        )
    )
    if userinput == "y":
        installpath = os.path.join(cosmicfishpie.__path__[0], "configs", "external_data")
    else:
        userinput = input("Do you want to install it to another path? (y/N)\n")
        if userinput == "y":
            installpath = input("Please give us the path to install to\n")
        else:
            exit()
    if not os.path.isdir(installpath):
        try:
            os.makedirs(installpath)
        except FileExistsError:
            pass

    userinput = input("Do you want to install the Euclid luminosity ratio file? (y/N)\n")
    if userinput == "y":
        _install_euclid_lumratio(installpath)
    userinput = input("Do you want to install the SKA external survey files? (y/N)\n")
    if userinput == "y":
        _install_ska_data(installpath)
    print("Done!")

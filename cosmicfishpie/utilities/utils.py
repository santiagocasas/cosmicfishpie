import math
import os
import subprocess
from collections import abc
from configparser import ConfigParser

import numpy as np


class misc:
    @staticmethod
    def deepupdate(original, update):
        """Recursively update a dict.

        Subdicts won't be overwritten but also updated.
        """
        if not isinstance(original, abc.Mapping):
            return update
        for key, value in update.items():
            if isinstance(value, abc.Mapping):
                original[key] = misc.deepupdate(original.get(key, {}), value)
            else:
                original[key] = value
        return original


class printing:
    debug = False

    @staticmethod
    def debug_print(*args, **kwargs):
        if printing.debug:
            print(*args, **kwargs)
        return None

    @staticmethod
    def time_print(
        feedback_level=0,
        min_level=0,
        text="Computation done in: ",
        time_ini=None,
        time_fin=None,
        instance=None,
    ):
        if time_ini is not None and time_fin is not None:
            elapsed_time = time_fin - time_ini
            ela_str = "  {:.2f} s".format(elapsed_time)
        else:
            ela_str = ""
        if instance is not None:
            instr = "In class: " + instance.__class__.__name__
        else:
            instr = ""
        if feedback_level >= min_level:
            print("")
            print(instr + "  " + text + ela_str)
        return None


class numerics:
    old_round_decimals_up = False

    @staticmethod
    def moving_average(data_set, periods=2):
        weights = np.ones(periods) / periods
        return np.convolve(data_set, weights, mode="valid")

    @staticmethod
    def round_decimals_up(number, decimals: int = 2, precision=5):
        """
        Returns a value rounded up to a specific number of decimal places.
        """
        if numerics.old_round_decimals_up:
            number = np.float16(np.format_float_positional(number, precision=precision))
            # this function protects from precision issues with floats
            if decimals == 0:
                return np.ceil(number)
            elif number < 1e-2:
                decimals = 4
            elif number < 1e-1:
                decimals = 3

            factor = 10**decimals
            rounded_number = np.ceil(number * factor) / factor
        else:
            exponent = math.floor(math.log10(number))
            mantissa = number / (10**exponent)
            rounded_mantissa = math.ceil(mantissa * 10) / 10
            rounded_number = rounded_mantissa * (10**exponent)
        return rounded_number

    @staticmethod
    def closest(lst, K):
        lst = np.asarray(lst)
        idx = (np.abs(lst - K)).argmin()
        return lst[idx]

    @staticmethod
    def bisection(array, value):
        """Given an ``array``, and given a ``value``, returns an
        index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing.
        Returns 0 if value is less than or equal to the first element,
        and len(array)-1 if value is greater than or equal to the last element.
        """
        array = np.asarray(array)  # Ensure array is a numpy array
        n = len(array)
        if np.isclose(value, array[0]) or value < array[0]:
            return 0
        if np.isclose(value, array[-1]) or value > array[-1]:
            return n - 2  # used with bins, which are defined as [z_i, z_i+1]

        left, right = 0, n - 1
        while left < right:
            mid = (left + right) // 2
            if np.isclose(array[mid], value):
                return mid
            elif array[mid] < value:
                left = mid + 1
            else:
                right = mid

        return left - 1

    @staticmethod
    def find_nearest(a, a0):
        "Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return idx


class filesystem:
    @staticmethod
    def mkdirp(dirpath):
        """
        This function creates the directory dirpath if it is not found
        :param dirpath: string with the path of the directory to be created
        :return:  None
        :rtype:   NoneType
        """
        outdir = os.path.dirname(dirpath)
        # create directory if it does not exist
        if outdir == "":
            print("Output root is on working directory")
        elif not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
                print((str(outdir) + " directory created"))
            except OSError:
                raise
        else:
            print((str(outdir) + "  exists already"))
        return None

    # Return the git revision as a string
    @staticmethod
    def git_version():
        # From: https://stackoverflow.com/a/40170206/1378746
        def _minimal_ext_cmd(cmd):
            # construct minimal environment
            env = {}
            for k in ["SYSTEMROOT", "PATH"]:
                v = os.environ.get(k)
                if v is not None:
                    env[k] = v
            # LANGUAGE is used on win32
            env["LANGUAGE"] = "C"
            env["LANG"] = "C"
            env["LC_ALL"] = "C"
            out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
            return out

        try:
            out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
            GIT_REVISION = out.strip().decode("ascii")
        except OSError:
            GIT_REVISION = "Unknown"
        except BaseException:
            GIT_REVISION = "Not a git repository"

        return GIT_REVISION


class inputiniparser:
    def __init__(
        self,
        config_dir,
        config_file="main.ini",
        parameter_translator=dict(),
        fiducial_translator=dict(),
    ):
        self.configmain = ConfigParser()
        self.configmain.optionxform = str
        self.configmain.read(config_dir + config_file)
        self.main_input_dir = config_dir
        print("Sections: ", self.configmain.sections())
        pars_var_dict = dict(self.configmain.items("params_varying"))
        translator = parameter_translator
        pars_cosmo_dict = dict(self.configmain.items("params_cosmo"))
        self.main_paramnames = list(pars_var_dict.keys())
        self.mirror_paramnames = dict(zip(self.main_paramnames, self.main_paramnames))
        print("-- paramnames")
        print(self.main_paramnames)
        self.main_foldernames = [pars_var_dict[ii] for ii in self.main_paramnames]
        print("-- foldernames")
        print(self.main_foldernames)
        self.main_fiducials = [
            fiducial_translator.get(ii, 1.0) * float(pars_cosmo_dict[ii])
            for ii in self.main_paramnames
        ]
        print("-- fiducials")
        print(self.main_fiducials)
        self.main_paramnames = [
            translator.get(kk, self.mirror_paramnames[kk]) for kk in self.main_paramnames
        ]
        self.main_fidu_dict = dict(zip(self.main_paramnames, self.main_fiducials))
        print(self.main_fidu_dict)
        files_quant_dict = dict(self.configmain.items("output_files"))
        print("-- input files")
        self.main_filesquant_dict = dict()
        for kk in files_quant_dict:
            self.main_filesquant_dict[kk] = files_quant_dict[kk].replace(".txt", "")
        print(self.main_filesquant_dict)

        return None

    def free_epsilons(self, epsilon=0.01):
        main_precision_epsilons = self.configmain.get(
            "params_precision", "abs_epsilons"
        )  # list[main_precision_dict['abs_epsilons']]
        self.main_epsilons = [float(ss.strip(" ")) for ss in main_precision_epsilons.split(",")]
        print("varying epsilons")
        print(self.main_epsilons)
        if epsilon not in self.main_epsilons:
            print("Warning:  Provided epsilon is not in list of varying epsilons")
        self.main_freepars_dict = dict(
            zip(self.main_paramnames, len(self.main_paramnames) * [epsilon])
        )
        return None


class physmath:
    sr = np.power(180 / np.pi, 2)

    @staticmethod
    def areasky():
        fullarea = 4 * np.pi * physmath.sr
        return fullarea

    @staticmethod
    def radtodeg(rads):
        degs = rads * physmath.sr
        return degs

    @staticmethod
    def degtorad(degs):
        rads = degs / physmath.sr
        return rads

    def sqdegtofsky(sqd):
        fsky = sqd / physmath.areasky()
        return fsky

    def fskytosqdeg(fsky):
        sqdeg = fsky * physmath.areasky()
        return sqdeg

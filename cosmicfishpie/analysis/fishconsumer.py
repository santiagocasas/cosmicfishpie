import collections
import copy
import os
import re

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from chainconsumer import Chain, ChainConfig, ChainConsumer, PlotConfig, Truth

import cosmicfishpie.analysis.colors as fc
import cosmicfishpie.analysis.fisher_plot_analysis as fpa
from cosmicfishpie.utilities.utils import printing as upr

dprint = upr.debug_print

colarray = [[str(ii), matplotlib.colors.to_hex(fc.nice_colors(ii))] for ii in [0, 1, 2, 3, 4, 5, 6]]


usercolors_barplot = [
    ["purple", "#8338ec"],
    ["cyan", "#3a86ff"],
    ["orange", "#fb5607"],
    ["red", "#d11149"],
    ["yellow", "#ffbe0b"],
]

usercolors_darker = [
    ["darkpurple", "#420d8c"],
    ["darkcyan", "#003b99"],
    ["darkorange", "#7e2902"],
    ["darkred", "#760a28"],
    ["darkyellow", "#805e00"],
]

colors = [
    ("Red", "#FF0000"),
    ("Green", "#00FF00"),
    ("Blue", "#0000FF"),
    ("Yellow", "#FFFF00"),
    ("Purple", "#800080"),
    ("Orange", "#FFA500"),
    ("Teal", "#008080"),
    ("Pink", "#FFC0CB"),
    ("Brown", "#A52A2A"),
    ("Grey", "#808080"),
]

allnicecolors = usercolors_barplot + usercolors_darker + colarray + colors
allnicecolors = [tuple(item) for item in allnicecolors]
allnicecolors_dict = dict(allnicecolors)
allnicecolors_list = list(allnicecolors_dict.values())
allnicecolors_namelist = list(allnicecolors_dict.keys())

barplot_filter_names = [
    "GC_{sp} opt",
    "WL opt",
    "GC_{sp}+WL opt",
    "WL+GC_{ph}+XC_{ph} opt",
    "GC_{sp}+WL+GC_{ph}+XC_{ph} opt",
]


def display_colors(colors, figsize=(6, 6)):
    """
    Display a pie chart of colors with their names and hex codes.

    This function creates a visual representation of a color palette by generating
    a pie chart where each slice represents a color. Each color is labeled with its
    name and hex code. The labels are positioned around the pie chart for clear visibility.

    Parameters:
        colors (list of tuples): A list of tuples containing the name and hex code
                                for each color. Each tuple should be in the format
                                (name, hex_code), where name is a string and hex_code
                                is a string representing a valid hex color (e.g., "#FF0000").
        figsize (tuple, optional): The figure size in inches as (width, height).
                                  Defaults to (6, 6).

    Returns:
        None: The function displays the pie chart but does not return any value.

    Example:
        >>> colors = [("Red", "#FF0000"), ("Green", "#00FF00"), ("Blue", "#0000FF")]
        >>> display_colors(colors)
    """

    # Create a list of color hex codes
    color_codes = [color[1] for color in colors]

    # Create a list of color names
    color_names = [str(color[0]) + ":" + str(color[1]) for color in colors]

    # Create a pie chart
    fig, ax = plt.subplots(figsize=figsize)
    wedges, _ = ax.pie([1] * len(color_codes), colors=color_codes)

    # Add labels for each color
    for i, color in enumerate(color_names):
        angle = (wedges[i].theta2 + wedges[i].theta1) / 2.0
        x = 1.2 * np.cos(np.deg2rad(90 - angle))
        y = 1.2 * np.sin(np.deg2rad(90 - angle))
        ha = "center" if x == 0 else ("left" if x > 0 else "right")
        va = "center" if y == 0 else ("top" if y > 0 else "bottom")
        print(color)
        rotation_angle = 0.1 * i
        # if  angle > 90:
        #    rotation_angle = angle - 90
        # if angle > 180 :
        #    rotation_angle = angle -180
        # else:
        #    rotation_angle = angle
        x1, y1 = wedges[i].center
        x2, y2 = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        dx, dy = 1.2 * np.array([x2, y2]) / np.sqrt(x2**2 + y2**2)
        ax.annotate(
            color,
            xy=(x1, y1),
            xytext=(x1 + dx, y1 + dy),
            ha=ha,
            va=va,
            color=color_codes[i],
            rotation=rotation_angle,
            fontsize=10,
            # =rotation_angle,
            # arrowprops=dict(arrowstyle='->', color='k', connectionstyle='arc3,rad=0'),
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
        )
    # Set the title
    # plt.title('Color Pie Chart')

    # Display the chart
    plt.show()


def clamp(x):
    """
    Clamp a value between 0 and 255.

    This function ensures a value is within the valid range for RGB color components.

    Parameters:
        x (int or float): The value to clamp

    Returns:
        int: The clamped value between 0 and 255
    """
    return max(0, min(x, 255))


def hex2rgb(hexcode):
    """
    Convert a hexadecimal color code to RGB tuple.

    Parameters:
        hexcode (str): A hexadecimal color code string (e.g., "#FF0000")

    Returns:
        tuple: An RGB tuple with values from 0-255 (e.g., (255, 0, 0))
    """
    return tuple(map(ord, hexcode[1:].decode("hex")))


def rgb2hex(rgb_tuple):
    """
    Convert an RGB tuple to a hexadecimal color code.

    Parameters:
        rgb_tuple (tuple): An RGB tuple with values typically between 0-1 or 0-255

    Returns:
        str: A hexadecimal color code string (e.g., "#FF0000")
    """
    return mcolors.to_hex(rgb_tuple)
    # r, g, b = rgb_tuple
    # return "#{:02x}{:02x}{:02x}".format(clamp(r),clamp(g), clamp(b))


def add_mathrm(s):
    # The regex pattern will match any sequence of two or more alphanumeric characters
    pattern = r"[a-zA-Z0-9]{2,}"

    # Find all the substrings that match the pattern
    substrings = re.findall(pattern, s)

    # Replace the matched substrings with the surrounded string
    for substring in substrings:
        s = s.replace(substring, r"\mathrm{" + substring + "}")

    # Replace spaces with LaTeX proper space '\,'
    s = s.replace(" ", r"\,\,")

    # Surround the final string with dollar signs
    s = f"${s}$"

    return s


def replace_latex_name(fisher_matrix, old_str, new_str):
    old_list = fisher_matrix.get_param_names_latex()
    new_list = [new_str if ii == old_str else ii for ii in old_list]
    fisher_matrix.set_param_names_latex(new_list)


def replace_latex_style(fmat, replace_dict):
    oldpartex = fmat.get_param_names_latex()
    newpartex = [pp.replace(pp, replace_dict.get(pp, pp)) for pp in oldpartex]
    fmat.set_param_names_latex(newpartex)
    return fmat


def fishtable_to_pandas(
    paramstab,
    fishAnalysis,
    default_titles=["Relative 1sigma errors: ", "Absolute 1sigma errors: "],
    title="",
    set_titles=False,
    apply_formats=False,
    filter_names=None,
    return_data_bar=False,
):
    """
    Converts a fisher matrix table to pandas DataFrames.
    This function takes a list of parameters and a FisherAnalysis object, and returns two pandas DataFrames:
    one for relative errors (as percentages) and one for absolute errors. It can also return data for bar plots.
    Parameters
    ----------
    paramstab : list
        List of parameter names to include in the table.
    fishAnalysis : FisherAnalysis
        The FisherAnalysis object containing the fisher matrices.
    default_titles : list, optional
        Default titles for the tables. Default is ["Relative 1sigma errors: ", "Absolute 1sigma errors: "].
    title : str, optional
        Additional title to append to the default titles. Default is "".
    set_titles : bool, optional
        If True, set captions for the DataFrames. Default is False.
    apply_formats : list or False, optional
        If provided, should be a list of two format strings to apply to the relative and absolute DataFrames.
        Default is False.
    filter_names : list or None, optional
        List of fisher matrix names to filter when returning bar plot data. Default is None.
    return_data_bar : bool, optional
        If True, return data formatted for bar plots instead of DataFrames. Default is False.
    Returns
    -------
    tuple or dict
        If return_data_bar is False, returns a tuple of two pandas DataFrames (fish_df, fish_df_abs).
        If return_data_bar is True, returns a dictionary with keys as formatted fisher names and values as relative errors.
    """

    fishers_totable_marg = fishAnalysis.marginalise(paramstab, update_names=False)
    fishers_totable_marg_list = fishers_totable_marg.get_fisher_list()
    table_data = collections.OrderedDict()
    table_abs_data = collections.OrderedDict()
    barplot_data = collections.OrderedDict()
    for ff in fishers_totable_marg_list:
        table_data["fiducials"] = ff.param_fiducial
        rel_perc_err = np.zeros(len(ff.param_fiducial))
        rel_err = np.zeros(len(ff.param_fiducial))
        for ii, fifi in enumerate(ff.param_fiducial):
            if fifi != 0:
                rel_err[ii] = np.abs((ff.get_confidence_bounds()[ii]) / fifi)
                rel_perc_err[ii] = 100 * rel_err[ii]
            else:
                rel_err[ii] = np.abs(ff.get_confidence_bounds()[ii])
                rel_perc_err[ii] = rel_err[ii]
        table_data[ff.name] = rel_perc_err
        abs_err = np.abs(ff.get_confidence_bounds())
        table_abs_data["fiducials"] = ff.param_fiducial
        table_abs_data[ff.name] = abs_err
        if return_data_bar:
            if ff.name in filter_names:
                newname = ff.name.split(" ")[0]
                # print(newname)
                ltxname = "$\\mathrm{" + str(newname).replace(" ", "\\ ") + "}$"
                barplot_data[ltxname] = rel_err
    if return_data_bar:
        return barplot_data
    df_titles = [default_titles[ii] + title for ii in range(len(default_titles))]
    fish_df = pd.DataFrame.from_dict(table_data, orient="index", columns=paramstab)
    fish_df_abs = pd.DataFrame.from_dict(table_abs_data, orient="index", columns=paramstab)
    if apply_formats:
        fish_df = fish_df.applymap(apply_formats[0].format)
        fish_df_abs = fish_df_abs.applymap(apply_formats[1].format)
    if set_titles:
        fish_df = fish_df.style.set_caption(df_titles[0])
        fish_df_abs = fish_df_abs.style.set_caption(df_titles[1])
    return fish_df, fish_df_abs


def customize_barh(
    data,
    width_bar=1,
    width_space=0.5,
    lims1=[0, 150],
    cols_dict=None,
    filename="forecast",
    parslist=None,
    data_title="Legend",
    xlabel=r"$\sigma/\theta_{\rm fid}$",
    legend_cols=2,
    ylabels_fontsize=32,
    tickspacing=[1],
    savefig=False,
    outpath="./",
    dpi=100,
):
    n_measure = len(data)  # number of measure per people
    n_people = data[list(data.keys())[0]].size  # number of people

    almbck = "k"  #'#333333' #'dimgray'#'#262626'almbck  #almost black
    none = "None"  #'dimgray'#'#262626'almbck  #almost black

    # some calculation to determine the position of Y ticks labels
    total_space = n_people * (n_measure * width_bar) + (n_people - 1) * width_space
    ind_space = n_measure * width_bar
    step = ind_space / 2.0
    pos = np.arange(step, total_space + width_space, ind_space + width_space)

    # create the figure and the axes to plot the data
    fig, ax = plt.subplots(1, sharey=True, figsize=(14, 9), facecolor="w")
    #    fig, ax = plt.figure(1,figsize=(10,8), facecolor='w')
    # remove top and right spines and turn ticks off if no spine
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position("bottom")
    # postition of tick out

    lwax = 1.5
    ax.spines["left"].set_linewidth(lwax)
    ax.spines["bottom"].set_linewidth(lwax)
    ax.spines["top"].set_linewidth(lwax)
    ax.spines["left"].set_color(almbck)
    ax.spines["bottom"].set_color(almbck)
    ax.spines["top"].set_color(almbck)

    lwbar = 1.0

    # plot the data
    for i, keyword in enumerate(data.keys()):
        ax.barh(
            pos - step + i * width_bar,
            data[keyword],
            width_bar,
            facecolor=cols_dict[keyword],
            edgecolor=none,
            linewidth=lwbar,
            label=keyword,
        )

    plt.setp(ax.get_yticklabels(), visible=True)

    ax.set_yticks(pos)
    # you may want to use the list of name as argument of the function to be more
    # flexible (if you have to add a people)
    ax.set_yticklabels(parslist, fontsize=ylabels_fontsize, rotation=45)
    ax.set_ylim((-width_space, total_space + width_space))
    plt.setp(ax.get_yticklabels(), visible=True)

    lgd = ax.legend(
        loc="upper right",  # , bbox_to_anchor=(legend_anchorbox[0],legend_anchorbox[1]),
        ncol=legend_cols,
        fancybox=False,
        shadow=False,
        fontsize=26,
        title=data_title,
        frameon=True,
        title_fontsize=28,
    )

    plt.setp(lgd.get_title(), fontsize=24)
    ax.set_xlim(lims1[0], lims1[1])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tickspacing[0]))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(tickspacing[0] / 2.0))
    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.1, 1])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.xaxis.set_ticklabels(["0.001", "0.01", "0.1", "1"], fontsize=22)
    # ax.xaxis.set_ticklabels(['0.1%', '1%', '10%', '100%'], fontsize=20)
    ax.set_xlabel(xlabel, fontsize=ylabels_fontsize + 2)

    tickw = 1.0
    ax.tick_params(
        which="major",
        axis="both",
        direction="in",
        width=tickw,
        length=6,
        labelsize=ylabels_fontsize,
        pad=8,
        colors=almbck,
        labelcolor=almbck,
        right="off",
        left="on",
    )

    ax.tick_params(
        which="minor",
        axis="both",
        direction="in",
        width=tickw,
        length=3,
        labelsize=12,
        pad=8,
        colors=almbck,
        labelcolor=almbck,
        right="off",
        left="on",
    )
    # plt.subplots_adjust(top=0.65)
    ax.yaxis.set_ticks_position("left")  # ticks position on the right
    # plt.subplots_adjust(bottom=0.13)
    # plt.subplots_adjust(left=0.23)
    # plt.subplots_adjust(right=0.95)
    # fig.text(0.6, 0.01, r'$\sigma/\theta_{\rm fid}$', ha='center', size=25, color=almbck)
    # plt.tight_layout(w_pad=5)
    # plt.savefig(filename+'.png', figsize=(8, 6), dpi=100,pad_inches=0.1,bbox_inches="tight")
    if savefig:
        print("Figure saved to: ", outpath + filename, ".pdf")
        plt.savefig(outpath + filename + ".pdf", bbox_inches="tight", dpi=dpi)
    # plt.show()


def perc_to_abs(perc_sig, fid):
    sigma_abs = np.abs(perc_sig / 100 * fid)
    return sigma_abs


def log_fidu_to_fidu(logfid):
    fid = np.power(10, logfid)
    return fid


def sigma_fidu(log_fidu, sigma_perc_log, sign):
    sigma_log = perc_to_abs(sigma_perc_log, log_fidu)
    fidu = log_fidu_to_fidu(log_fidu)
    sigma_fid = sign * (np.power(10, log_fidu + sign * sigma_log) - fidu)
    return sigma_fid


def gaussian(x, mu, sigma):
    pref = 1 / (sigma * np.sqrt(2 * np.pi))
    expo = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return pref * expo


def n_sigmas(log_fidu, sigma_log, nsigmas=1):
    fidu = log_fidu_to_fidu(log_fidu)
    val_plus = fidu + nsigmas * sigma_fidu(log_fidu, sigma_log, +1)
    val_minus = fidu - nsigmas * sigma_fidu(log_fidu, sigma_log, -1)
    return [val_minus, val_plus]


def arrays_gaussian(log_fidu, perc_sigma, nsigmas=1):
    sigma_log = perc_to_abs(perc_sigma, log_fidu)
    xa = np.linspace(log_fidu - nsigmas * sigma_log, log_fidu + nsigmas * sigma_log, 1000)
    g1d = gaussian(xa, log_fidu, sigma_log)
    return xa, g1d


def prepare_fishers(
    tupcases, fishers_database_dict, colors_todisplay=usercolors_barplot, display_namedcolors=False
):
    fishcasestr = "--"
    Ntup = len(tupcases)
    for tup in tupcases:
        fishcasestr += "_".join(tup) + "_"
    print("plot fishcase string: ", fishcasestr)
    fisher_case_analdict = {}
    for ii, tup in enumerate(tupcases):
        fisher_case_analdict[ii] = copy.deepcopy(fishers_database_dict[tup])
    ellipses_names_dic = {}
    for ii in range(Ntup):
        print(tupcases[ii])
        ellipses_names_dic[ii] = [ff.name for ff in fisher_case_analdict[ii].get_fisher_list()]
        for jj, fn in enumerate(ellipses_names_dic[ii]):
            print(jj, " : ", fn)
        print("---")
    ret_dic = {
        "tupcases": tupcases,
        "ellipses_names_dic": ellipses_names_dic,
        "fisher_case_analdict": fisher_case_analdict,
        "fishcasestr": fishcasestr,
    }
    if display_namedcolors:
        display_colors(colors_todisplay)
    else:
        print(colors_todisplay)
    return ret_dic


param_latex_names = [
    "\\Omega_m",
    "\\Omega_b",
    "h",
    "n_s",
    "\\log(f_{R0})",
    "log10fR0",
    "\\sigma_{8}",
    "\\sigma_8",
]
parnames_style = [
    "\\Omega_{{\\rm m,0}}",
    "\\Omega_{{\\rm b,0}}",
    "h",
    "n_{\\rm s}",
    "\\log_{10}\\left|f_{R0}\\right|",
    "\\log_{10}\\left|f_{R0}\\right|",
    "\\sigma_{8}",
    "\\sigma_{8}",
]
dictreplace_tex = dict(zip(param_latex_names, parnames_style))


def choose_fish_toplot(
    indices_to_fish=[[(0, "1")]],
    pars_toplot=["Omegam"],
    colors_toplot=None,
    tupcases=None,
    ellipses_names_dic=None,
    fisher_case_analdict=None,
    fisher_group=None,
    fishcasestr=None,
    fisher_labels=None,
    marginalise=True,
    reshuffle=False,
    texify_names=False,
    named_colors_list=allnicecolors,
    texreplace_dict=dictreplace_tex,
    rename_fishers=False,
):
    fishers_toplot = fpa.CosmicFish_FisherAnalysis()
    cols_dict = dict(named_colors_list)
    cols_toplot = []
    colnames_toplot = []
    if fisher_case_analdict is not None and fisher_group is None:
        Ntups = len(tupcases)
        print("N cases: ", Ntups)
        for ii in range(Ntups):
            tup = tupcases[ii]
            fish_ana = copy.deepcopy(fisher_case_analdict[ii])
            # print('ind', indices_to_fish)
            kk = indices_to_fish[ii]
            # print('kk', kk)
            for gg in kk:
                nn = ""
                # print('gg', gg)
                if len(gg) > 1:
                    jj = gg[0]
                    cc = gg[1]
                    if len(gg) > 2:
                        nn = gg[2]
                elif len(gg) == 1:
                    jj = gg[0]
                    cc = named_colors_list[ii * jj][0]
                ff = fish_ana.get_fisher_list()[jj]
                # print("old name: ", ff.name)
                if rename_fishers:
                    newname = (ff.name).split(" ")[0] + " " + tup[0] + " " + tup[1] + " " + nn
                    ff.name = newname
                    # print("new name of Fisher: ", ff.name)
                ff = replace_latex_style(ff, texreplace_dict)
                fishers_toplot.add_fisher_matrix(ff)
                # print(ff.get_param_names())
                colnames_toplot.append(cc)
                cols_toplot.append(
                    cols_dict.get(cc, "k")
                )  # get color from named list, otherwise black
    elif fisher_case_analdict is None and fisher_group is not None:
        for ii, fifi in enumerate(fisher_group.fisher_list):
            ff = replace_latex_style(fifi, texreplace_dict)
            fishers_toplot.add_fisher_matrix(ff)
            if colors_toplot is None:
                cols_toplot.append(list(cols_dict.values())[ii])
                colnames_toplot.append(list(cols_dict.keys())[ii])
            else:
                cols_toplot.append(colors_toplot[ii])
                colnames_toplot.append(str(colors_toplot[ii]))
    if marginalise:
        fishers_toplot_m = fishers_toplot.marginalise(pars_toplot, update_names=False)
    elif reshuffle:
        fishers_toplot_m = fishers_toplot.reshuffle(pars_toplot, update_names=False)
    else:
        fishers_toplot_m = fishers_toplot
    flabs = []
    print("*** Fishers and params for plotting")
    for ii, fish in enumerate(fishers_toplot_m.get_fisher_list()):
        print("Fisher for plot: ", f"{ii}: ", fish.name)
        if fisher_labels is not None:
            flabs.append(fisher_labels[ii])
        else:
            if texify_names:
                flabs.append(add_mathrm(fish.name))
            else:
                flabs.append(fish.name)
        print("Fisher label name: ", flabs[ii])
        print("Fisher color name={:s} , code={:s}: ".format(colnames_toplot[ii], cols_toplot[ii]))
        print("Parameter names: ", fish.get_param_names())
        print("Latex par names: ", fish.get_param_names_latex())
        print("Fiducials: ", fish.get_param_fiducial())
        parnames_latex = ["${:s}$".format(nam) for nam in fish.get_param_names_latex()]
        fish.set_param_names_latex(parnames_latex)
        fb = fish.get_confidence_bounds()
        print("Confidence Bounds: ")
        print(fb)
        if ii == 0:
            f0bounds = fish.get_confidence_bounds()
        if ii > 0:
            print("Ratio of bounds to fisher 0: ")
            Nb = np.min([len(f0bounds), len(fb)])
            ## protection against params of different size
            fbratio = fb[:Nb] / f0bounds[:Nb]
            print(["{:.2f}".format(fbi) for fbi in fbratio])
        print("____")
    retdic = {
        "tupcases": tupcases,
        "ellipses_names_dic": ellipses_names_dic,
        "fisher_case_analdict": fisher_case_analdict,
        "fishcasestr": fishcasestr,
    }
    retdic["fishers_toplot_group"] = fishers_toplot_m
    retdic["fisher_labels"] = flabs
    retdic["pars_toplot"] = pars_toplot
    retdic["cols_toplot"] = cols_toplot
    retdic["texreplace_dict"] = texreplace_dict
    # retdic.update(locals())
    return retdic


def prepare_settings_plot(
    retdic,
    nsigmas_contour=3,
    sigmas_zoom_factor_dict={"sigma8": [1.15, 1.15], "h": [0.1, 0.1]},
    which_fish=0,
    plot_filename=None,
):
    fishcasestr = retdic["fishcasestr"]
    if fishcasestr is None:
        fishcasestr = ""
    fish_labs = retdic["fisher_labels"]
    if plot_filename is None:
        plotstri = "_".join(fish_labs) + "--" + fishcasestr
        plotstri = (
            plotstri.replace(" ", "")
            .replace("{", "_")
            .replace("}", "_")
            .replace("__", "_")
            .replace("+", "")
        )
        plotstri = (
            plotstri.replace(r"__", "")
            .replace(r"\,", "-")
            .replace("$", "")
            .replace(r"\\", "")
            .replace("\\mathrm", "")
            .replace("___", "_")
            .replace("__", "_")
            .replace("----", "-")
            .replace("--", "-")
        )
    else:
        plotstri = str(plot_filename)
    print(f" File name for plot: {plotstri}")
    fisher_toplot_list = retdic["fishers_toplot_group"].get_fisher_list()
    pars_to_plot = retdic["pars_toplot"]
    # for ii, fish0 in fisher_toplot_list:
    extentdic = {}
    extentos = []
    nsigmas = nsigmas_contour
    extradic = sigmas_zoom_factor_dict
    fish0 = fisher_toplot_list[which_fish]
    for pp in pars_to_plot:
        try:
            ii = fish0.get_param_index(pp)
            fidii = fish0.get_param_fiducial()[ii]
            sigmaii = fish0.get_confidence_bounds()[ii]
            ex = extradic.get(pp, [1, 1])
            exmin = ex[0]
            exmax = ex[1]
            print(pp, exmin, exmax)
            extentdic[pp] = [fidii - nsigmas * sigmaii * exmin, fidii + nsigmas * sigmaii * exmax]
            print(extentdic[pp])
            extentos.append(extentdic[pp])
        except KeyError:
            pass
    retdic["extents"] = extentos
    retdic["parnames_latex"] = fish0.get_param_names_latex()
    retdic["parnames"] = fish0.get_param_names()
    retdic["parnames_to_tex"] = dict(zip(retdic["parnames"], retdic["parnames_latex"]))
    # retdic['parnames_to_tex'].update(retdic['texreplace_dict'])
    retdic["par_fiducials"] = fish0.get_param_fiducial()
    retdic["plot_filename"] = plotstri
    return retdic


def make_triangle_plot(
    fishers=None,
    chains=None,
    fisher_labels=None,
    chain_labels=None,
    params=None,
    colors=None,
    truth_values=None,
    shade_fisher=False,
    shade_chains=True,
    ls_fisher="-",
    lw_fisher=2.5,
    legend_kwargs={"fontsize": 20, "loc": "upper right"},
    param_labels: dict = {},
    label_font_size=20,
    tick_font_size=16,
    smooth=3,
    kde=False,
    bins=None,
    extents: dict = {},
    figsize=None,
    transform_params: dict = None,
    shade_alpha=0.4,
    save_plot=False,
    savepath="./",
    plot_filename=None,
    file_format=".pdf",
    save_dpi=200,
    savefile=None,
):
    """Create a triangle plot from Fisher matrices and/or MCMC chains using ChainConsumer.

    Parameters
    ----------
    fishers : list, optional
        List of Fisher matrix objects with attributes param_fiducial, fisher_matrix_inv, param_names
    chains : list, optional
        List of pandas DataFrames containing MCMC chains with 'weight' column
    fisher_labels : list, optional
        Labels for Fisher matrices in the plot legend
    chain_labels : list, optional
        Labels for MCMC chains in the plot legend
    params : list, optional
        Parameters to plot. If None, uses all parameters from first Fisher/chain
    colors : list, optional
        Colors for each Fisher/chain. Defaults to a preset color scheme
    truth_values : dict, optional
        Dictionary of true parameter values to plot as vertical lines
    shade_fisher : bool or list, optional
        Whether to shade Fisher contours. Can be single bool or list for each Fisher
    shade_chains : bool or list, optional
        Whether to shade chain contours. Can be single bool or list for each chain
    ls_fisher : str or list, optional
        Line style for Fisher contours. Can be single str or list for each Fisher
    lw_fisher : float or list, optional
        Line width for Fisher contours. Can be single float or list for each Fisher
    fontsize : int, optional
        Font size for legend and labels
    param_labels : dict, optional
        Dictionary mapping parameter names to LaTeX labels. Default provides common cosmological parameters
    smooth : int, optional
        Smoothing factor for contours
    kde : bool, optional
        Whether to use KDE for histograms
    bins : int, optional
        Number of bins for histograms
    extents : list or dict, optional
        Parameter ranges to plot, as [min, max] for each parameter
    figsize : tuple, optional
        Figure size as (width, height) in inches
    transform_params : dict, optional
        Dictionary to transform parameters, with required keys:
            'param': parameter name to transform
            'transform_param': name of transformed parameter
            'transform_func': function to apply to parameter samples
        And optional keys:
            'transform_param_latex': latex name of transformed parameter
    shade_alpha : float, optional
        Alpha (transparency) for contour shading
    save_plot : bool, optional
        Whether to save the plot
    savepath : str, optional
        Directory to save plot
    plot_filename : str, optional
        Filename for saved plot (without extension)
    file_format : str, optional
        File format for saved plot (default: '.pdf')
    save_dpi : int, optional
        DPI for saved plot
    savefile : str, optional
        Complete filepath to save plot (overrides savepath, plot_filename, and file_format)

    Returns
    -------
    matplotlib.figure.Figure
        The generated triangle plot

    Examples
    --------
    >>> # Plot just Fisher matrices
    >>> fig = make_triangle_plot(
    ...     fishers=[fisher1, fisher2],
    ...     fisher_labels=['SKAO', 'Euclid']
    ... )

    >>> # Plot Fisher matrices and chains
    >>> fig = make_triangle_plot(
    ...     fishers=[fisher1],
    ...     chains=[chain_df],
    ...     fisher_labels=['Fisher'],
    ...     chain_labels=['MCMC'],
    ...     truth_values={'Omegam': 0.3, 'h': 0.7}
    ... )

    >>> # Plot with parameter transformation
    >>> fig = make_triangle_plot(
    ...     fishers=[fisher1],
    ...     fisher_labels=['Fisher'],
    ...     transform_params={
    ...         'param': 'sigma8',
    ...         'transform_param': 'S8',
    ...         'transform_param_latex': r'$S_8$',
    ...         'transform_func': lambda x: x * np.sqrt(0.3/x),
    ...     }
    ... )
    """
    # Initialize ChainConsumer
    c = ChainConsumer()

    # Default colors
    default_colors = ["#3a86ff", "#fb5607", "#8338ec", "#ffbe0b", "#d11149"]
    if colors is None:
        colors = default_colors

    # Default parameter labels
    default_param_labels = {
        "Omegam": r"$\Omega_{{\rm m}, 0}$",
        "Omegab": r"$\Omega_{{\rm b}, 0}$",
        "h": r"$h$",
        "ns": r"$n_{\rm s}$",
        "sigma8": r"$\sigma_8$",
        "bI_c1": r"$bI_{{\rm c}, 1}$",
        "bI_c2": r"$bI_{{\rm c}, 2}$",
    }
    if param_labels == {}:
        param_labels = default_param_labels

    # Manage parameters to transform
    def identity(x):
        return x

    transform_func = identity
    orig_param = None
    new_param = None

    if transform_params is not None:
        if not all(k in transform_params for k in ["param", "transform_param", "transform_func"]):
            raise ValueError(
                "transform_params must contain 'param', 'transform_param', and 'transform_func' keys"
            )

        transform_func = transform_params["transform_func"]
        orig_param = transform_params["param"]
        new_param = transform_params["transform_param"]
        new_param_latex = transform_params.get("transform_param_latex")
        if new_param_latex and new_param not in param_labels:
            param_labels[new_param] = new_param_latex

    # Process Fisher matrices
    if fishers is not None:
        if fisher_labels is None:
            fisher_labels = [f"Fisher {i+1}" for i in range(len(fishers))]

        # Convert shade_fisher to list if needed
        if isinstance(shade_fisher, bool):
            shade_fisher = [shade_fisher] * len(fishers)

        # Convert ls_fisher to list if needed
        if isinstance(ls_fisher, str):
            ls_fisher = [ls_fisher] * len(fishers)

        # Convert lw_fisher to list if needed
        if isinstance(lw_fisher, (int, float)):
            lw_fisher = [lw_fisher] * len(fishers)

        for i, fisher in enumerate(fishers):
            # Get parameter names and possibly transform them
            param_names = fisher.param_names

            # Handle parameter transformation if requested
            if transform_params is not None and orig_param in param_names:
                # Create a copy of parameter names and fiducials
                param_names = list(param_names)
                fiducial_values = list(fisher.param_fiducial)

                # Find index of parameter to transform
                idx = param_names.index(orig_param)

                # Transform fiducial value
                fiducial_values[idx] = transform_func(fiducial_values[idx])

                # Replace parameter name
                param_names[idx] = new_param

                # Create transformed covariance matrix
                cov = fisher.inverse_fisher_matrix()

                fishchain = Chain.from_covariance(
                    mean=fiducial_values,
                    covariance=cov,
                    columns=param_names,
                    color=colors[i % len(colors)],
                    linestyle=ls_fisher[i] if isinstance(ls_fisher, list) else ls_fisher,
                    linewidth=lw_fisher[i] if isinstance(lw_fisher, list) else lw_fisher,
                    shade=shade_fisher[i],
                    name=fisher_labels[i],
                )
            else:
                # Use standard Chain.from_covariance without transformation
                fishchain = Chain.from_covariance(
                    mean=fisher.param_fiducial,
                    covariance=fisher.inverse_fisher_matrix(),
                    columns=fisher.param_names,
                    color=colors[i % len(colors)],
                    linestyle=ls_fisher[i] if isinstance(ls_fisher, list) else ls_fisher,
                    linewidth=lw_fisher[i] if isinstance(lw_fisher, list) else lw_fisher,
                    shade=shade_fisher[i],
                    name=fisher_labels[i],
                )
            c.add_chain(fishchain)

    # Process MCMC chains
    if chains is not None:
        if chain_labels is None:
            chain_labels = [f"Chain {i+1}" for i in range(len(chains))]

        # Convert shade_chains to list if needed
        if isinstance(shade_chains, bool):
            shade_chains = [shade_chains] * len(chains)

        start_color = len(fishers) if fishers is not None else 0
        for j, chain in enumerate(chains):
            # Filter chains with zero weight if weight column exists
            if "weight" in chain.columns:
                chain_nonzero = chain[chain["weight"] > 0]
            else:
                chain_nonzero = chain

            # Transform parameter if requested
            if transform_params is not None and orig_param in chain_nonzero.columns:
                chain_nonzero = chain_nonzero.copy()
                chain_nonzero[new_param] = transform_func(chain_nonzero[orig_param])

            c.add_chain(
                Chain(
                    samples=chain_nonzero,
                    name=chain_labels[j],
                    color=colors[(start_color + j) % len(colors)],
                    shade=shade_chains[j],
                )
            )

    # Add truth values if provided
    if truth_values is not None:
        # Transform truth value if needed
        if transform_params is not None and orig_param in truth_values:
            truth_copy = truth_values.copy()
            truth_copy[new_param] = transform_func(truth_values[orig_param])
            c.add_truth(Truth(location=truth_copy))
        else:
            c.add_truth(Truth(location=truth_values))

    # Configure plot settings

    c.set_plot_config(
        PlotConfig(
            sigma2d=False,
            summary=True,
            plot_point=True,
            legend_kwargs=legend_kwargs,
            labels=param_labels,
            label_font_size=label_font_size,
            tick_font_size=tick_font_size,
            figsize=figsize,
            extents=extents,
        )
    )

    c.set_override(ChainConfig(smooth=smooth, kde=kde, bins=bins, shade_alpha=shade_alpha))

    # Transform parameters list if needed
    if params is not None and transform_params is not None and orig_param in params:
        params = list(params)
        idx = params.index(orig_param)
        params[idx] = new_param

    # Create the plot
    fig = c.plotter.plot(columns=params)

    # Save the plot if requested
    if savefile is not None:
        fig.savefig(savefile, bbox_inches="tight", dpi=save_dpi)
        print(f"Plot saved to: {savefile}")
    elif save_plot:
        if plot_filename is None:
            plot_filename = "triangle_plot"
        full_path = f"{savepath}{plot_filename}{file_format}"
        fig.savefig(full_path, bbox_inches="tight", dpi=save_dpi)
        print(f"Plot saved to: {full_path}")

    return fig


def plot_chain_summary(
    chains,
    chain_names=None,
    truth_values=None,
    output_file=None,
    show_errorbars=True,
    linestyle="--",
    linecolor="black",
    blind_params=None,
    plot_config_kwargs={},
):
    """
    Create a summary plot of MCMC chains using ChainConsumer.

    This function provides a simpler interface for creating summary plots of chains,
    useful for visualizing parameter constraints in a compact format.

    Parameters
    ----------
    chains : list
        List of pandas DataFrames containing MCMC chains
    chain_names : list, optional
        List of names for each chain to display in the legend
    truth_values : dict, optional
        Dictionary of true parameter values to plot as vertical lines
    output_file : str, optional
        Path to save the output figure. If None, the figure is not saved
    show_errorbars : bool, optional
        Whether to show error bars on the summary plot (default: True)
    linestyle : str, optional
        Line style for truth values (default: '--')
    linecolor : str, optional
        Color for truth value lines (default: 'black')
    blind_params : list, optional
        List of parameter names to blind (exclude) from the plot

    Returns
    -------
    matplotlib.figure.Figure
        The generated summary plot figure

    Examples
    --------
    >>> fig = plot_chain_summary(
    ...     chains=[chain1, chain2],
    ...     chain_names=['MCMC Run 1', 'MCMC Run 2'],
    ...     truth_values=fiducial_values,
    ...     output_file='./results/chain_summary.png'
    ... )
    """
    # Initialize ChainConsumer
    c = ChainConsumer()

    # Add chains to ChainConsumer
    if not isinstance(chains, list):
        chains = [chains]

    # Set default chain names if not provided
    if chain_names is None:
        chain_names = [f"Chain {i+1}" for i in range(len(chains))]
    elif not isinstance(chain_names, list):
        chain_names = [chain_names]

    # Add each chain
    for i, chain in enumerate(chains):
        c.add_chain(Chain(samples=chain, name=chain_names[i]))

    # Add truth values if provided
    if truth_values is not None:
        c.add_truth(Truth(location=truth_values, linestyle=linestyle, color=linecolor))

    # Configure plot settings
    plot_config = PlotConfig(**plot_config_kwargs)

    # Handle blind parameters
    if blind_params:
        plot_config.blind = blind_params

    # Set the plot configuration
    c.set_plot_config(plot_config)

    # Create the plot
    fig = c.plotter.plot_summary(errorbar=show_errorbars)

    # Save the figure if output path is provided
    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Plot saved to: {output_file}")

    return c, fig


def load_Nautilus_chains_from_txt(filename, param_cols, log_weights=False):
    """Load Nautilus chains from a text file."""
    if isinstance(param_cols, dict):
        param_cols = list(param_cols.keys())
    elif not isinstance(param_cols, list):
        raise TypeError("param_cols must be a list or dict")
    chain_arr = np.loadtxt(filename)
    chain_df = pd.DataFrame(chain_arr, columns=param_cols + ["weight", "posterior"])
    if log_weights:
        chain_df["weight"] = np.exp(chain_df["weight"])
    chain_df = chain_df[chain_df["weight"] > 0]
    return chain_df


def parse_log_param(log_file_path):
    """Extract truth values from MontePython log.param file.

    Parameters
    ----------
    log_file_path : str
        Path to the log.param file

    Returns
    -------
    dict
        Dictionary with parameter names as keys and their truth values as values
    """
    truth_values = {}

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "data.parameters['" in line and "]" in line:
                    param_name = line.split("['")[1].split("']")[0]
                    values_str = line.split("=")[1].strip()
                    try:
                        values = eval(values_str)
                        truth_values[param_name] = values[0]
                    except Exception as e:
                        print(
                            f"Warning: Could not parse values for parameter {param_name}: {str(e)}"
                        )
    except Exception as e:
        raise ValueError(f"Error reading log.param file: {str(e)}")

    if not truth_values:
        raise ValueError("No parameter values found in log.param file")

    return truth_values


def load_montepython_chains(
    base_path,
    chain_root,
    num_chains,
    burn_in=0.3,
    param_names=None,
    param_names_conversion_dict=None,
    derived_params=None,
    chain_suffix=".txt",
    start_index=1,
):
    """Load and process multiple MontePython chains.

    Parameters
    ----------
    base_path : str
        Path to the folder containing chains and log.param
    chain_root : str
        Root name of the chain files (e.g., "2024-09-11_200000_")
    num_chains : int
        Number of chains to load
    burn_in : float, optional
        Fraction of initial chain to remove (default: 0.3)
    param_names : list, optional
        List of parameter names. If None, will try to read from '.paramnames' file
    param_conversion_dict : dict, optional
        Dictionary to convert parameter names (e.g., {'omega_cdm': 'wc'})
    derived_params : dict, optional
        Dictionary of derived parameters to compute. Each key is the new parameter name
        and the value is a tuple of (source_param, transform_function)
    chain_suffix : str, optional
        Suffix of chain files (default: '.txt')
    start_index : int, optional
        Starting index for chain numbering (default: 1)

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame containing all chains with burn-in removed and metadata
    """
    # Validate burn-in
    if not (0 <= burn_in < 1):
        raise ValueError("burn_in must be between 0 and 1")

    # Construct full paths
    chain_path = os.path.join(base_path, chain_root)
    param_file = os.path.join(base_path, chain_root + ".paramnames")
    log_param_file = os.path.join(base_path, "log.param")
    original_param_names = param_names
    # Read parameter names if not provided
    if param_names is None:
        try:
            original_param_names = np.genfromtxt(param_file, dtype=str, usecols=0)
            original_param_names = list(original_param_names)
        except Exception as e:
            raise ValueError(f"Could not read parameter names file: {str(e)}")

    # Read truth values
    try:
        truth_values = parse_log_param(log_param_file)
        print(f"Successfully read truth values for {len(truth_values)} parameters")
    except Exception as e:
        print(f"Warning: could not read truth values: {str(e)}")
        truth_values = None

    # Convert parameter names if dictionary provided
    if param_names_conversion_dict is not None:
        param_names = [param_names_conversion_dict.get(p, p) for p in original_param_names]
        # Also convert truth value keys if they exist
        if truth_values is not None:
            truth_values = {
                param_names_conversion_dict.get(k, k): v for k, v in truth_values.items()
            }
    else:
        param_names = original_param_names

    chain_dfs = []

    # Load and process each chain
    for ci in range(num_chains):
        # Construct chain path
        full_chain_path = f"{chain_path}_{ci + start_index}{chain_suffix}"
        print(f"Loading chain from {full_chain_path}")

        try:
            # Load chain
            chain_np = np.loadtxt(full_chain_path)

            # Create DataFrame
            chain_df = pd.DataFrame(chain_np, columns=["weight", "posterior"] + param_names)

            # Remove burn-in
            if burn_in > 0:
                start_idx = int(burn_in * len(chain_df))
                chain_df = chain_df.iloc[start_idx:]

            # Append to list
            chain_dfs.append(chain_df)

        except Exception as e:
            print(f"Warning: Failed to load chain {full_chain_path}: {str(e)}")

    if not chain_dfs:
        raise ValueError("No chains were successfully loaded")

    # Concatenate all chains
    combined_chain = pd.concat(chain_dfs, ignore_index=True)

    # Add metadata as attributes
    combined_chain.attrs["num_chains"] = len(chain_dfs)
    combined_chain.attrs["total_samples"] = len(combined_chain)
    combined_chain.attrs["samples_per_chain"] = [len(df) for df in chain_dfs]
    combined_chain.attrs["original_params"] = original_param_names
    combined_chain.attrs["params"] = param_names
    combined_chain.attrs["columns"] = list(combined_chain.columns)
    if truth_values is not None:
        combined_chain.attrs["truth_values"] = truth_values

    if derived_params is not None:
        for new_param, (source_param, transform_func) in derived_params.items():
            try:
                # Add derived parameter to chain
                combined_chain[new_param] = combined_chain[source_param].apply(transform_func)
                combined_chain.attrs["params"].append(new_param)
                combined_chain.attrs["columns"] = list(combined_chain.columns)
                # Add derived parameter to truth values
                if truth_values is not None and source_param in truth_values:
                    source_truth = truth_values[source_param]
                    derived_truth = transform_func(source_truth)
                    combined_chain.attrs["truth_values"][new_param] = derived_truth
                    print(f"Added truth value for derived parameter {new_param}: {derived_truth}")

            except Exception as e:
                print(f"Warning: Failed to compute derived parameter {new_param}: {str(e)}")

    return combined_chain


class FishConsumer:
    """Convenience wrapper that exposes the fishconsumer helpers as instance methods."""

    def __init__(self, named_colors=None, barplot_colors=None):
        self.named_colors = copy.deepcopy(named_colors or allnicecolors)
        self.barplot_colors = copy.deepcopy(barplot_colors or usercolors_barplot)

    def display_colors(self, colors=None, figsize=(6, 6)):
        color_list = colors if colors is not None else self.named_colors
        return display_colors(color_list, figsize=figsize)

    def clamp(self, x):
        return clamp(x)

    def hex2rgb(self, hexcode):
        return hex2rgb(hexcode)

    def rgb2hex(self, rgb_tuple):
        return rgb2hex(rgb_tuple)

    def add_mathrm(self, s):
        return add_mathrm(s)

    def replace_latex_name(self, fisher_matrix, old_str, new_str):
        return replace_latex_name(fisher_matrix, old_str, new_str)

    def replace_latex_style(self, fmat, replace_dict):
        return replace_latex_style(fmat, replace_dict)

    def fishtable_to_pandas(self, *args, **kwargs):
        return fishtable_to_pandas(*args, **kwargs)

    def customize_barh(self, data, **kwargs):
        if "cols_dict" not in kwargs or kwargs["cols_dict"] is None:
            kwargs["cols_dict"] = dict(self.barplot_colors)
        return customize_barh(data, **kwargs)

    def perc_to_abs(self, perc_sig, fid):
        return perc_to_abs(perc_sig, fid)

    def log_fidu_to_fidu(self, logfid):
        return log_fidu_to_fidu(logfid)

    def sigma_fidu(self, log_fidu, sigma_perc_log, sign):
        return sigma_fidu(log_fidu, sigma_perc_log, sign)

    def gaussian(self, x, mu, sigma):
        return gaussian(x, mu, sigma)

    def n_sigmas(self, log_fidu, sigma_log, nsigmas=1):
        return n_sigmas(log_fidu, sigma_log, nsigmas=nsigmas)

    def arrays_gaussian(self, log_fidu, perc_sigma, nsigmas=1):
        return arrays_gaussian(log_fidu, perc_sigma, nsigmas=nsigmas)

    def prepare_fishers(
        self,
        tupcases,
        fishers_database_dict,
        colors_todisplay=None,
        display_namedcolors=False,
    ):
        palette = (
            colors_todisplay if colors_todisplay is not None else copy.deepcopy(self.barplot_colors)
        )
        return prepare_fishers(
            tupcases,
            fishers_database_dict,
            colors_todisplay=palette,
            display_namedcolors=display_namedcolors,
        )

    def choose_fish_toplot(self, *args, **kwargs):
        if "named_colors_list" not in kwargs or kwargs["named_colors_list"] is None:
            kwargs["named_colors_list"] = copy.deepcopy(self.named_colors)
        return choose_fish_toplot(*args, **kwargs)

    def prepare_settings_plot(self, *args, **kwargs):
        return prepare_settings_plot(*args, **kwargs)

    def make_triangle_plot(self, *args, **kwargs):
        return make_triangle_plot(*args, **kwargs)

    def load_Nautilus_chains_from_txt(self, filename, param_cols, log_weights=False):
        return load_Nautilus_chains_from_txt(filename, param_cols, log_weights=log_weights)

    def parse_log_param(self, log_file_path):
        return parse_log_param(log_file_path)

    def load_montepython_chains(self, *args, **kwargs):
        return load_montepython_chains(*args, **kwargs)


DEFAULT_FISH_CONSUMER = FishConsumer()

import collections
import copy
import re

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from chainconsumer import ChainConsumer
from numpy.random import multivariate_normal

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
    Display a pie chart of colors with their names.

    Parameters:
        colors (list of tuples): A list of tuples containing the name and hex code for each color.

    Returns:
        None
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
    return max(0, min(x, 255))


def hex2rgb(hexcode):
    return tuple(map(ord, hexcode[1:].decode("hex")))


def rgb2hex(rgb_tuple):
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
                ltxname = "$\\mathrm{" + str(newname).replace(" ", "\ ") + "}$"
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
dictreplace_tex


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
            .replace("\mathrm", "")
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


def chainfishplot(
    return_dictionary,
    **cckwargs,
):
    """
    Chain fish plot function

    Parameters:

    return_dic: Dictionary containing quantities and settings to plot

    cckwargs:  Arguments for ChainConsumer and other optional arguments

        transform_chains: Dictionary to transform a single parameter in the chains
               Keys:
                    'param' : parameter name to transform
                    'transform_param_latex' : latex name of transformed parameter
                    'transform_param' : name of transformed parameter
                    'transform_func' : function (must be broadcastable) to be applied to the samples of that parameter
                    'transformed_extents': 'automatic'|[list]  New bounds for the transformed parameter range


    """
    c = ChainConsumer()
    return_dic = copy.deepcopy(return_dictionary)
    fishers_toplot_list = return_dic["fishers_toplot_group"].get_fisher_list()
    Nfishes = len(fishers_toplot_list)
    fisher_labels = return_dic["fisher_labels"]
    colors = return_dic["cols_toplot"]
    extents = return_dic["extents"]
    def_plotname = return_dic["plot_filename"]

    gaussian_samples = cckwargs.get("gaussian_samples", 100000)
    zorder_list = cckwargs.get("zorder_list", [None] * Nfishes)

    transform_chains = cckwargs.get("transform_chains", {})
    std_str = "Std."

    if transform_chains != {}:
        tr_func = transform_chains["transform_func"]
        par_tr = transform_chains["param"]
        tex_tr_par = transform_chains["transform_param_latex"]
        tr_par = transform_chains["transform_param"]
        p_ind = return_dic["pars_toplot"].index(par_tr)
        if transform_chains["transformed_extents"] == "automatic":
            extents[p_ind] = [
                tr_func(ee) for ee in extents[p_ind]
            ]  # due to chainconsumer bug, needs to be list, not np array
        else:
            extents[p_ind] = transform_chains["transformed_extents"]
        return_dic["pars_toplot"][p_ind] = tr_par
        return_dic["parnames_to_tex"][tr_par] = tex_tr_par
        std_str = "Trans."
        def_plotname = "transformed_" + str(tr_par) + "_" + return_dic["plot_filename"]
        print(f"{std_str} params to plot: ")
        print(return_dic["pars_toplot"])
        print(return_dic["parnames_to_tex"])

    for ii, ff in enumerate(fishers_toplot_list):
        print(ff.param_fiducial, ff.inverse_fisher_matrix())
        data = multivariate_normal(
            ff.param_fiducial, ff.inverse_fisher_matrix(), size=gaussian_samples
        )
        data_tr = np.copy(data)
        texpars = ff.get_param_names_latex()
        if transform_chains != {}:
            if par_tr in ff.get_param_names():
                par_ind = ff.get_param_index(par_tr)
                data_tr[:, par_ind] = tr_func(data_tr[:, par_ind])
                texpars[par_ind] = tex_tr_par
            else:
                print(f"Warning: transformed {par_tr} not in Fisher matrix")
        c.add_chain(data_tr, name=fisher_labels[ii], parameters=texpars, zorder=zorder_list[ii])
        print("-- Fisher name  |  Paramnames tex   |   {:s} Confidence Bounds --".format(std_str))
        print(ff.name, ff.get_param_names_latex(), ff.get_confidence_bounds())

    smooth = cckwargs.get("smooth", 3)
    shade_alpha = cckwargs.get("shade_alpha", 0.4)
    kde = cckwargs.get("kde", False)
    diagonal_tick_labels = cckwargs.get("diagonal_tick_labels", True)
    def_gradient = cckwargs.get("default_gradient", 0.2)
    shade_gradient = cckwargs.get("shade_gradient", [def_gradient] * Nfishes)
    def_leg_kw = {"loc": (1.5, 1), "fontsize": 20}
    legend_kwargs = cckwargs.get("legend_kwargs", def_leg_kw)
    def_shade = [True] * Nfishes
    shade = cckwargs.get("shade", def_shade)
    def_ls = Nfishes * ["-"]
    linestyles = cckwargs.get("linestyles", def_ls)
    linewidths = cckwargs.get("linewidths", 2.5)
    max_ticks = cckwargs.get("max_ticks", 2)
    tick_font_size = cckwargs.get("tick_font_size", 16)
    label_font_size = cckwargs.get("label_font_size", 20)
    c.configure(
        plot_hists=True,
        sigma2d=False,
        smooth=smooth,
        kde=kde,
        colors=colors,
        linewidths=linewidths,
        shade_gradient=shade_gradient,
        legend_kwargs=legend_kwargs,
        shade=shade,
        shade_alpha=shade_alpha,
        legend_color_text=True,
        legend_location=(1, 0),
        diagonal_tick_labels=diagonal_tick_labels,
        tick_font_size=tick_font_size,
        label_font_size=label_font_size,
        max_ticks=max_ticks,
        bar_shade=True,
        linestyles=linestyles,
    )

    plot_pars_names_in = cckwargs.get("plot_pars_names", return_dic["pars_toplot"])
    plot_pars_names = [return_dic["parnames_to_tex"].get(pp, pp) for pp in plot_pars_names_in]

    figsize = cckwargs.get("figsize", "page")
    legend = cckwargs.get("legend", True)
    print(plot_pars_names)
    fig = c.plotter.plot(
        parameters=plot_pars_names, figsize=figsize, legend=legend, extents=extents
    )

    # fig2 = c.plotter.plot_distributions(parameters=[r'$f_{R0}$'], display=True)

    special_axes = cckwargs.get("special_axes", dict())
    if special_axes != {}:
        print(fig.get_axes())
        axx = fig.get_axes()[special_axes["xindex"]]
        axy = fig.get_axes()[special_axes["yindex"]]
        lims = special_axes["lims"]
        axx.ticklabel_format(axis="x", style="sci", scilimits=(lims[0], lims[1]))
        axy.ticklabel_format(axis="y", style="sci", scilimits=(lims[0], lims[1]))
        xlabel = special_axes["xlabel"]
        ylabel = special_axes["ylabel"]
        axx.set_xlabel(xlabel)
        axy.set_ylabel(ylabel)

    savepath = cckwargs.get("savepath", "./")
    plot_filename = cckwargs.get("plot_filename", def_plotname)
    file_format = cckwargs.get("file_format", ".pdf")
    save_dpi = cckwargs.get("save_dpi", 200)
    save_plot = cckwargs.get("save_plot", True)

    plotfilename = savepath + plot_filename + file_format
    if save_plot:
        fig.savefig(plotfilename, dpi=save_dpi, bbox_inches="tight")
        print("Plot saved to: ", plotfilename)
    return fig

def simple_fisher_plot(
    fisher_list,
    params_to_plot,
    labels=None,
    colors=None,
    save_plot=False,
    legend=True,
    n_samples=10000,
    output_file="fisher_plot.pdf"
):
    """Create a triangle plot from Fisher matrices using ChainConsumer.
    
    Parameters
    ----------
    fisher_list : list
        List of CosmicFish_FisherMatrix objects to plot
    params_to_plot : list
        List of parameter names to include in the plot
    labels : list, optional
        Labels for each Fisher matrix in the legend
    colors : list, optional
        Colors for each Fisher matrix. Defaults to built-in colors
    save_plot : bool, optional
        Whether to save the plot to file (default: False)
    output_file : str, optional
        Filename for saving the plot (default: 'fisher_plot.pdf')
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated triangle plot figure
    """
    # Initialize ChainConsumer
    c = ChainConsumer()
    
    # Default colors if none provided
    if colors is None:
        colors = ['#3a86ff', '#fb5607', '#8338ec', '#ffbe0b', '#d11149']
        colors = colors[:len(fisher_list)]  # Truncate to needed length
    
    # Default labels if none provided
    if labels is None:
        labels = [f"Fisher {i+1}" for i in range(len(fisher_list))]
    
    # Generate samples for each Fisher matrix
    n_samples = 100000
    for i, fisher in enumerate(fisher_list):
        # Get samples from multivariate normal using Fisher matrix
        samples = multivariate_normal(
            fisher.param_fiducial,
            fisher.inverse_fisher_matrix(),
            size=n_samples
        )
        
        # Add chain to plot
        c.add_chain(
            samples,
            parameters=fisher.get_param_names(),
            name=labels[i],
            color=colors[i]
        )
    
    # Configure plot settings
    c.configure(
        plot_hists=True,
        sigma2d=False,
        smooth=3,
        colors=colors,
        shade=True,
        shade_alpha=0.3,
        bar_shade=True,
        linewidths=2,
        legend_kwargs={"fontsize": 12}
    )
    
    # Create the plot
    fig = c.plotter.plot(parameters=params_to_plot, legend=legend)
    
    # Save if requested
    if save_plot:
        fig.savefig(output_file, bbox_inches='tight', dpi=200)
        print(f"Plot saved to: {output_file}")
        
    return fig

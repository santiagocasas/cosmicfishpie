"""
   :synopsis: Module for creating comparison plots of Fisher Matrix entries and related visualizations.
   :module author: Dida Markovic, Santiago Casas, and other contributors to the CosmicFishPie project.
"""

import os

import matplotlib
import matplotlib.patches as mpatches

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from mpl_toolkits import axes_grid1

from cosmicfishpie.analysis import fisher_operations as fo
from cosmicfishpie.analysis import utilities as fu
from cosmicfishpie.utilities.utils import printing as upr

dprint = upr.debug_print

plt.style.use("tableau-colorblind10")

snscolors = sns.color_palette("colorblind")
marks = ["o", "s", "*", "X", "p", "^", "v", "8", "D", "P"]
# Colorblind list is 10 elements long, as well as markers list
# This will crash if more than 10 Fishers to compare are requested

matplotlib.rcParams["savefig.transparent"] = False
matplotlib.rcParams["figure.autolayout"] = True
# matplotlib.rcParams['xtick.major.pad']     = 4.0
# matplotlib.rcParams['ytick.major.pad']     = 4.0
# matplotlib.rcParams['axes.labelpad']       = 5.0
# matplotlib.rcParams['savefig.dpi']         = 300
# matplotlib.rcParams['savefig.pad_inches']  = 0.0
# use latex for all text handling. The following fonts
matplotlib.rcParams["text.usetex"] = False
# matplotlib.rcParams['text.latex.preamble'] =
# '\usepackage{amsmath},\usepackage{pgfplots},\usepackage[T1]{fontenc}'
# If True (default), the text will be antialiased.
matplotlib.rcParams["text.antialiased"] = True
# This only affects the Agg backend.


def calc_y_range(axis, yrang=None):
    yymin, yymax = axis.get_ylim()
    yymax = np.max(np.abs([yymin, yymax]))
    if yymax < 0.1:
        yymax = 0.1
        locat = 0.01
    if yymax < 0.3:
        yymax = 0.3
        locat = 0.05
    elif yymax < 1.0:
        yymax = 1.0
        locat = 0.1
    elif yymax < 5.0:
        yymax = 5.0
        locat = 1.0
    elif yymax < 10.0:
        yymax = 10.0
        locat = 5.0
    else:
        yymax = 1.05 * yymax
        locat = yymax // 5
    yymin = -yymax
    if yrang is not None:
        locat = (np.abs(np.max(yrang)) + np.abs(np.min(yrang))) / 5
        yymax = np.max(yrang)
        yymin = np.min(yrang)
    return (yymin, yymax, locat)


def og_plot_shades(
    ax,
    x_arr,
    x_names,
    lighty_arr=None,
    darky_arr=None,
    mats_labels=None,
    lightdark_names=["marg.", "unmarg."],
    cols=[],
    plotdark=True,
    plotlight=True,
    yrang=None,
    x_limpad=0.2,
    fish_leg_loc="upper left",
    LW=2,
    colordark="darkgrey",
    colorlight="lightgrey",
    alpha=0.7,
    light_hatch="/",
    patches_legend_loc="upper right",
    patches_legend_fontsize=16,
    dots_legend_fontsize=20,
    ylabelfontsize=20,
    ncol_legend=None,
    colors=None,
    color_palette='colorblind',
    legend_title_fontsize=None,
    legend_title=None,
    y_label="Differences",  # r'% differences on ' +r'$\sigma_i$'
    yticklabsize=18,
    xticklabsize=18,
    xtickfontsize=15,
    xticksrotation=0,
):
    LW = LW
    colD = colordark  # 'lightslategray'
    colL = colorlight
    aalpha = alpha
    darkgreypatch = mpatches.Patch(color=colD, alpha=aalpha)
    lightgreypatch = mpatches.Patch(color=colL, alpha=aalpha, hatch=light_hatch)
    if colors is None:
        colors = sns.color_palette(color_palette)
    if lighty_arr is not None:
        max_l = np.max(lighty_arr, 0)
        min_l = np.min(lighty_arr, 0)
    if darky_arr is not None:
        max_d = np.max(darky_arr, 0)
        min_d = np.min(darky_arr, 0)

    if plotlight and lighty_arr is None:
        print("Error: plotlight is True but lighty_arr is None. Aborting.")
        return None
    if plotdark and darky_arr is None:
        print("Error: plotdark is True but darky_arr is None. Aborting.")
        return None

    numarrs = len(mats_labels)
    nc = ncol_legend
    if ncol_legend is None:
        if numarrs < 6:
            nc = numarrs
        elif numarrs >= 6:
            nc = numarrs // 2

    for ii, lbl in enumerate(mats_labels):
        if plotlight:
            ax.plot(
                x_arr, lighty_arr[ii, :], marks[ii], c=colors[ii], ms=LW * 8, alpha=aalpha, label=lbl
            )
        if plotdark:
            if not plotlight:
                ms = LW * 8
                lbl = lbl
            else:
                ms = 0
                lbl = None
            ax.plot(
                x_arr,
                darky_arr[ii, :],
                marks[ii],
                c=colors[ii],
                ms=ms,
                mew=2,
                alpha=aalpha,
                label=lbl,
            )
    if plotlight:
        ax.fill_between(
            x_arr,
            min_l,
            max_l,
            interpolate=True,
            facecolor=colL,
            edgecolor=colL,
            alpha=aalpha,
            linewidth=0.0,
            hatch=light_hatch,
        )
    if plotdark:
        ax.fill_between(
            x_arr,
            min_d,
            max_d,
            interpolate=True,
            facecolor=colD,
            edgecolor=colD,
            alpha=aalpha,
            linewidth=0.0,
        )
    patchlist = []
    legpatch = []
    if plotlight:
        patchlist.append(lightgreypatch)
        legpatch.append(lightdark_names[0])
    if plotdark:
        patchlist.append(darkgreypatch)
        legpatch.append(lightdark_names[1])
    if plotlight or plotdark:
        leg2 = ax.legend(
            patchlist, legpatch, loc=patches_legend_loc, ncol=2, fontsize=patches_legend_fontsize
        )

    ax.legend(
        loc=fish_leg_loc,
        ncol=nc,
        fontsize=dots_legend_fontsize,
        handlelength=2,
        numpoints=1,
        title=legend_title,
        title_fontsize=legend_title_fontsize,
    )
    # bbox_to_anchor=(1.05, 1.05),
    ax.add_artist(leg2)
    # ax.axhline(y=0.0, ls=':', c='k', alpha=0.2) # we don't want the zero-line
    ax.set_ylabel(y_label, labelpad=1, fontsize=ylabelfontsize)
    ax.set_xlim([min(x_arr) - x_limpad, max(x_arr) + x_limpad])
    ax.tick_params(axis="x", direction="in", pad=10, labelsize=xticklabsize)
    ax.tick_params(axis="y", direction="in", pad=10, labelsize=yticklabsize)
    ax.set_xticks(x_arr)
    ax.set_xticklabels(x_names, fontsize=xtickfontsize, rotation=xticksrotation)
    ax.yaxis.tick_left()
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.tick_bottom()
    ax.xaxis.set_ticks_position("both")

    ymin, ymax, locaty = calc_y_range(ax, yrang)
    ax.set_ylim([ymin, ymax])
    majorLocator = ticker.MultipleLocator(locaty)
    minorLocator = ticker.MultipleLocator(locaty)
    majorFormatter = ticker.FormatStrFormatter("%.2f")
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)

    return ax


def plot_shades(
    ax,
    x_arr,
    x_names,
    lighty_arr=None,
    darky_arr=None,
    mats_labels=None,
    lightdark_names=["marg.", "unmarg."],
    plotdark=True,
    plotlight=True,
    yrang=None,
    x_limpad=0.2,
    fish_leg_loc="upper left",
    LW=2,
    colordark="darkgrey",
    colorlight="lightgrey",
    alpha=0.7,
    light_hatch="/",
    patches_legend_loc="upper right",
    patches_legend_fontsize=16,
    dots_legend_fontsize=20,
    ylabelfontsize=20,
    ncol_legend=None,
    colors=None,
    color_palette='colorblind',
    legend_title_fontsize=None,
    legend_title=None,
    y_label="Differences",  # r'% differences on ' +r'$\sigma_i$'
    yticklabsize=18,
    xticklabsize=18,
    xtickfontsize=15,
    xticksrotation=0,
):
    LW = LW
    # plt.style.use('tableau-colorblind10')
    colD = colordark  # 'lightslategray'
    colL = colorlight
    aalpha = alpha
    darkgreypatch = mpatches.Patch(color=colD, alpha=aalpha)
    lightgreypatch = mpatches.Patch(color=colL, alpha=aalpha, hatch=light_hatch)
    if colors is None:
        colors = sns.color_palette(color_palette)
    if lighty_arr is not None:
        max_l = np.max(lighty_arr, 0)
        min_l = np.min(lighty_arr, 0)
    if darky_arr is not None:
        max_d = np.max(darky_arr, 0)
        min_d = np.min(darky_arr, 0)

    if plotlight and lighty_arr is None:
        print("Error: plotlight is True but lighty_arr is None. Aborting.")
        return None
    if plotdark and darky_arr is None:
        print("Error: plotdark is True but darky_arr is None. Aborting.")
        return None

    numarrs = len(mats_labels)
    nc = ncol_legend
    if ncol_legend is None:
        if numarrs < 6:
            nc = numarrs
        elif numarrs >= 6:
            nc = numarrs // 2

    if plotlight:
        dprint("plotting light")
        ax.bar(x_arr, max_l, color=colL, width=0.8, alpha=0.9, zorder=1)
        ax.bar(x_arr, min_l, color=colL, width=0.8, alpha=0.9, zorder=1)
        # ax.fill_between(x_arr, min_l, max_l, interpolate=True, facecolor=colL,
        # edgecolor=colL, alpha=aalpha, linewidth=0.0, hatch=light_hatch)
    if plotdark:
        dprint("plotting dark")
        ax.bar(x_arr, max_d, color=colD, width=0.5, alpha=0.95, zorder=2)
        ax.bar(x_arr, min_d, color=colD, width=0.5, alpha=0.95, zorder=2)
        # ax.fill_between(x_arr, min_d, max_d, interpolate=True, facecolor=colD,
        # edgecolor=colD, alpha=aalpha, linewidth=0.0)
    for ii, lbl in enumerate(mats_labels):
        if plotlight:
            ax.scatter(
                x_arr,
                lighty_arr[ii, :],
                color=colors[ii],
                marker=marks[ii],
                s=(LW * 8) ** 2,
                label=lbl,
                alpha=aalpha,
                zorder=3 + ii,
            )
        if plotdark:
            if not plotlight:
                ax.scatter(
                    x_arr,
                    darky_arr[ii, :],
                    color=colors[ii],
                    marker=marks[ii],
                    s=(LW * 8) ** 2,
                    label=lbl,
                    alpha=aalpha - 0.1,
                    zorder=3 + ii,
                )
            else:
                lbl = None
    patchlist = []
    legpatch = []
    if plotlight:
        patchlist.append(lightgreypatch)
        legpatch.append(lightdark_names[0])
    if plotdark:
        patchlist.append(darkgreypatch)
        legpatch.append(lightdark_names[1])
    if plotlight or plotdark:
        leg2 = ax.legend(
            patchlist, legpatch, loc=patches_legend_loc, ncol=2, fontsize=patches_legend_fontsize
        )

    leg1 = ax.legend(
        loc=fish_leg_loc,
        ncol=nc,
        fontsize=dots_legend_fontsize,
        handlelength=2,
        numpoints=1,
        title=legend_title,
        title_fontsize=legend_title_fontsize,
    )
    # bbox_to_anchor=(1.05, 1.05),
    ax.add_artist(leg2)
    leg1.set_zorder(10)
    leg2.set_zorder(10)
    # ax.axhline(y=0.0, ls=':', c='k', alpha=0.2) # we don't want the zero-line
    ax.set_ylabel(y_label, labelpad=1, fontsize=ylabelfontsize)
    ax.set_xlim([min(x_arr) - x_limpad, max(x_arr) + x_limpad])
    ax.tick_params(axis="x", direction="in", pad=10, labelsize=xticklabsize)
    ax.tick_params(axis="y", direction="in", pad=10, labelsize=yticklabsize)
    ax.set_xticks(x_arr)
    ax.set_xticklabels(x_names, fontsize=xtickfontsize, rotation=xticksrotation)
    ax.yaxis.tick_left()
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.tick_bottom()
    ax.xaxis.set_ticks_position("both")

    ymin, ymax, locaty = calc_y_range(ax, yrang)
    ax.set_ylim([ymin, ymax])
    majorLocator = ticker.MultipleLocator(locaty)
    minorLocator = ticker.MultipleLocator(locaty)
    majorFormatter = ticker.FormatStrFormatter("%.2f")
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)

    return ax


def process_fish_errs(
    fishers_list,
    fishers_name,
    parstoplot=None,
    parsnames_latex=None,
    marginalize_pars=True,
    print_errors=True,
    compare_to_index=False,
    transform_latex_dict=dict(),
):
    # Cycle through files and get the errors and the present parameters
    print(("Fishers names: ", fishers_name))
    for nn, ff in zip(fishers_name, fishers_list):
        ff.name = nn

    if parstoplot is None:
        parstoplot = fishers_list[0].get_param_names()
    print(("parameters to plot: ", parstoplot))

    n_pars = len(parstoplot)
    x_pars = np.arange(1, n_pars + 1)

    # marginalize Fishers over parameters not plotted
    if marginalize_pars:
        processed_fishers = [fo.marginalise(ff, parstoplot) for ff in fishers_list]
    else:
        processed_fishers = [fo.reshuffle(ff, parstoplot) for ff in fishers_list]

    if parsnames_latex is None:
        parsnames_latex = processed_fishers[0].get_param_names_latex()
        # print(parsnames_latex)
        parsnames_latex_transf = [transform_latex_dict.get(pp, pp) for pp in parsnames_latex]
        print("X tick labels ---> :  ", parsnames_latex_transf)
        parsnames_latex = ["$" + pp + "$" for pp in parsnames_latex_transf]

    errMargs = np.array([mm.get_confidence_bounds(marginal=True) for mm in processed_fishers])
    errUnmargs = np.array([mm.get_confidence_bounds(marginal=False) for mm in processed_fishers])

    if print_errors:
        for ii, fishy in enumerate(processed_fishers):
            dprint(("Fisher name: ", fishy.name))
            dprint(("Parameter names latex: ", parsnames_latex))
            dprint(("Marginalized 1-sigma errors :", errMargs[ii]))
            dprint(("Unmarginalized 1-sigma errors :", errUnmargs[ii]))
    # Plot differences, not absolute values np.abs,   np.median default
    if not compare_to_index:
        eurel = fu.rel_median_error(errUnmargs)
        emrel = fu.rel_median_error(errMargs)
    else:
        if isinstance(compare_to_index, int) and compare_to_index >= 0:
            eurel = fu.rel_error_to_index(compare_to_index, errUnmargs)
            emrel = fu.rel_error_to_index(compare_to_index, errMargs)

    return eurel, emrel, x_pars, parsnames_latex


def ploterrs(
    fishers_list,
    fishers_name,
    parstoplot=None,
    parsnames_latex=None,
    marginalize_pars=True,
    plot_style="original",
    outpathfile=os.getcwd(),
    plot_marg=True,
    plot_unmarg=True,
    yrang=None,
    figsize=(10, 6),
    fish_leg_loc="lower left",
    dpi=400,
    savefig=True,
    y_label="Errors",
    ncol_legend=None,
    colors=None,
    legend_title_fontsize=None,
    legend_title=None,
    yticklabsize=20,
    xticklabsize=15,
    patches_legend_fontsize=20,
    dots_legend_fontsize=20,
    xtickfontsize=18,
    ylabelfontsize=20,
    compare_to_index=False,
    xticksrotation=0,
    save_error=False,
    transform_latex_dict=dict(),
    figure_title="",
):
    fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=figsize, facecolor="white")
    """ Plot the error comparison between different Fisher matrices"""
    ax1.set_title(figure_title, loc="center")
    eurel, emrel, x_pars, parsnames_latex = process_fish_errs(
        fishers_list,
        fishers_name,
        parstoplot=parstoplot,
        parsnames_latex=parsnames_latex,
        marginalize_pars=marginalize_pars,
        transform_latex_dict=transform_latex_dict,
        compare_to_index=compare_to_index,
    )
    # fishnamesjoined=("-").join(fishers_name)

    if save_error:
        np.savetxt(outpathfile.replace(".pdf", ".txt"), np.concatenate((eurel, emrel), axis=0))
    if plot_style == "original":
        og_plot_shades(
            ax1,
            x_pars,
            parsnames_latex,
            mats_labels=fishers_name,
            lighty_arr=emrel,
            darky_arr=eurel,
            lightdark_names=["marg.", "unmarg."],
            plotlight=plot_marg,
            plotdark=plot_unmarg,
            fish_leg_loc=fish_leg_loc,
            yrang=yrang,
            y_label=y_label,
            ncol_legend=ncol_legend,
            legend_title_fontsize=legend_title_fontsize,
            legend_title=legend_title,
            yticklabsize=yticklabsize,
            xticklabsize=xticklabsize,
            xtickfontsize=xtickfontsize,
            ylabelfontsize=ylabelfontsize,
            xticksrotation=xticksrotation,
            colors=colors,
            patches_legend_fontsize=patches_legend_fontsize,
            dots_legend_fontsize=dots_legend_fontsize,
        )
    elif plot_style == "bars":
        plot_shades(
            ax1,
            x_pars,
            parsnames_latex,
            mats_labels=fishers_name,
            lighty_arr=emrel,
            darky_arr=eurel,
            lightdark_names=["marg.", "unmarg."],
            plotlight=plot_marg,
            plotdark=plot_unmarg,
            fish_leg_loc=fish_leg_loc,
            yrang=yrang,
            y_label=y_label,
            ncol_legend=ncol_legend,
            legend_title_fontsize=legend_title_fontsize,
            legend_title=legend_title,
            yticklabsize=yticklabsize,
            xticklabsize=xticklabsize,
            xtickfontsize=xtickfontsize,
            ylabelfontsize=ylabelfontsize,
            xticksrotation=xticksrotation,
            colors=colors,
            patches_legend_fontsize=patches_legend_fontsize,
            dots_legend_fontsize=dots_legend_fontsize,
        )

    # fig.tight_layout(pad=10.0, w_pad=10.0, h_pad=10.0)
    plotfile = outpathfile
    if savefig:
        fig.savefig(plotfile, dpi=dpi, bbox_inches="tight")
    # fig.show()


def add_colorbar(im, aspect=30, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def matrix_plot(
    matrix,
    xlabel="Ratio",
    ticklabels=None,
    filename="matrixplot.png",
    figsize=(9, 9),
    colormap=plt.cm.viridis,
    savefig=True,
    dpi=200,
):
    fig, ax = plt.subplots(1, figsize=(9, 9), facecolor="white")
    intermat = matrix
    lenmat = intermat.shape[0]
    if ticklabels is None:
        ticklabels = ["{:d}".format(ii) for ii in range(lenmat)]

    for i in range(len(intermat)):
        for j in range(len(intermat)):
            c = intermat[j, i]
            ax.text(i, j, "{:.2f}".format(c), va="center", ha="center", fontsize=11)

    im = ax.matshow(intermat, cmap=plt.cm.viridis)
    # cax = fig.add_axes([ax.get_position().x1+0.06, ax.get_position().y0, 0.02, ax.get_position().height])
    # plt.colorbar(im, cax=cax)
    add_colorbar(im)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_position("top")

    matplotlib.rcParams["xtick.labelsize"] = 14
    matplotlib.rcParams["ytick.labelsize"] = 14
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=14,
        labelbottom=True,
        bottom=True,
        top=True,
        labeltop=False,
        direction="in",
    )

    ax.set_xticks(np.arange(lenmat), ticklabels)
    ax.set_yticks(np.arange(lenmat), ticklabels)
    if savefig:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

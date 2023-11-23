import os
import matplotlib.pyplot as plt
import matplotlib

from . import fisher_matrix as fm
from . import fisher_plot_analysis as fpa
from . import plot_comparison as pc

from cosmicfishpie.utilities.utils import filesystem as ffs

from getdist import plots
from getdist.gaussian_mixtures import GaussianND

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'font.family': 'cm'})
params = {'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'mathtext.bf': 'serif:bold',
          'mathtext.it': 'serif:italic',
          'mathtext.sf': 'sans\\-serif',
          'text.usetex': False,
          'font.family': 'serif',
          'font.weight': 'normal',
          'font.serif': 'Computer Modern'}

matplotlib.rcParams.update(params)
matplotlib.rcParams['text.usetex']


class fisher_plotting:
    """
    This class uses the cosmicfish_pylib classes to generate
    contour plots using getdist
    """

    def __init__(self, **options):

        self.options = options
        self.fish_files = options.get('fish_files', None)
        self.fishers_group = options.get('fishers_group', None)
        self.fishers_list = options.get('fishers_list', None)
        self.fish_labels = options.get('fish_labels', None)
        self.plot_pars = options['plot_pars']
        self.plot_method = options['plot_method']
        self.outroot = options.get('outroot', None)
        self.outpath = options.get(
            'outpath', os.path.join(
                os.getcwd(), 'plots'))
        self.axis_limits = options.get('axis_custom_factors', None)
        self.colors = options['colors']
        self.file_format = options.get('file_format', '.pdf')
        self.fishers_dict = dict()
        ffs.mkdirp(self.outpath)
        if self.fish_files is not None:
            print("Reading Fishers")
            self.fishers_group = self.read_fisher_matrices()
        elif self.fishers_list is not None:
            for fmat in self.fishers_list:
                if not isinstance(fmat, fm.fisher_matrix):
                    raise TypeError(
                        "Fisher matrix in list is not of the correct type fm.fishermatrix")
            if self.fish_labels is None:
                self.fish_labels = [fmat.name for fmat in self.fishers_list]
            self.fishers_group = fpa.CosmicFish_FisherAnalysis()
            for fmat in self.fishers_list:
                self.fishers_group.add_fisher_matrix(fmat)
        if self.fishers_group is not None:
            if not isinstance(
                    self.fishers_group,
                    fpa.CosmicFish_FisherAnalysis):
                raise TypeError(
                    "Loaded Fisher group is not of the correct type")
            if self.fish_labels is None:
                self.fish_labels = self.fishers_group.fisher_name_list
            for flab, fishm in zip(
                    self.fish_labels, self.fishers_group.fisher_list):
                print("Fisher matrix loaded, label name: ", flab)
                fishm.name = flab

        if self.fishers_group is None:
            raise ValueError("No Fisher matrices were loaded correctly")

        # if options['plot_method'] == 'Gaussian':
        #    self.plot_fisher(options)
        # else:
        #    raise ValueError("/!\ Unknown plot method {}. /!\ ".format(options['plot_method']))

    def read_fisher_matrices(self):
        self.fishers_group = fpa.CosmicFish_FisherAnalysis()
        for ffil, flab in zip(self.fish_files, self.fish_labels):
            fishm = fm.fisher_matrix(file_name=ffil)
            fishm.name = flab
            print("Fisher matrix file imported: ", ffil)
            print("Fisher matrix loaded, label name: ", flab)
            self.fishers_group.add_fisher_matrix(fishm)
        return self.fishers_group

    def get_FoM(self, ind):

        print('')
        print('Computing FoM...')
        print('')
        if all(x in self.fidpars[ind] for x in ['w0', 'wa']):
            print('WILL COMPUTE FOM')
        else:
            print('w0 and wa not in parameter list')
            print('no FoM computed')

    def load_gaussians(self):

        self.gaussians = []
        for ii, fishm in enumerate(self.fishers_group.fisher_list):
            # covariance = self.get_marginv([par for par in self.fidpars[ind]],ind).values
            invcov = fishm.fisher_matrix
            means = fishm.get_param_fiducial()
            print("---> Fisher matrix name: ", fishm.name)
            print("Fisher matrix fiducials: \n", means)
            bounds = fishm.get_confidence_bounds()
            print("Fisher matrix 1-sigma bounds: \n", bounds)
            self.param_names = fishm.get_param_names()
            print("Fisher matrix param names: \n", self.param_names)
            self.param_labels = fishm.get_param_names_latex()
            print("Fisher matrix param names latex: \n", self.param_labels)
            # print(labels)
            self.gaussians.append(GaussianND(means, invcov,
                                             is_inv_cov=True,
                                             names=self.param_names,
                                             labels=self.param_labels))
            if ii == 0:
                means_0 = means
                self.paramnames_0 = self.param_names
                bounds_0 = bounds
                self.fiducial_markers = dict()
                self.param_bounds_0 = dict()
                for pp, par in enumerate(self.paramnames_0):
                    self.fiducial_markers[par] = means_0[pp]
                    self.param_bounds_0[par] = bounds_0[pp]

        return self.gaussians

    def param_limits_bounds(self, axis_custom_factors=None):
        factors_def = dict()
        for par in self.paramnames_0:
            factors_def[par] = 2.0
        if axis_custom_factors is not None:
            for kk in axis_custom_factors.keys():
                factors_def[kk] = axis_custom_factors[kk]
            # print(factors_def)
        elif self.axis_limits:
            self.axis_limits['all'] = self.axis_limits.get('all', None)
            # print('here')
            if self.axis_limits['all']:
                for par in self.plot_pars:
                    factors_def[par] = self.axis_limits['all']
            for key in self.axis_limits.keys():
                factors_def[key] = self.axis_limits[key]
            # print(factors_def)
        centers = self.fiducial_markers
        onesigmas = self.param_bounds_0
        self.param_lims_bounds = dict()
        for par in self.paramnames_0:
            self.param_lims_bounds[par] = [
                centers[par] - onesigmas[par] * factors_def[par],
                centers[par] + onesigmas[par] * factors_def[par]]
        print(self.param_lims_bounds)
        return self.param_lims_bounds

    def plot_fisher(self, **kwargs):
        """
        Generates a triangle plot based on loaded gaussian data and specified parameters.

        Parameters:
            **kwargs: Keyword arguments for customizing the plot.
                axis_custom_factors (optional): Custom factors for axis limits. Default is None.
                filled (optional): Boolean value indicating whether contour plots should be filled or not. Default is True.
                contour_args (optional): List of dictionaries specifying contour plot arguments. Default is [{'alpha':0.9}].
                legend_loc (optional): Location of the legend in the plot. Default is 'upper right'.
                dpi (optional): Dots per inch for saving the plot to a file. Default is 300.
                file_format (optional): File format for saving the plot. Default is '.pdf'.
                marker_color (optional): Color of the axis markers. Default is 'black'.
                axes_fontsize (optional): Font size for the axes labels. Default is 20.
                legend_fontsize (optional): Font size for the legend labels. Default is 20.
                figure_legend_frame (optional): Frame thickness for the figure legend. Default is 20.
                axes_labelsize (optional): Font size for the axes tick labels. Default is 20.
                figure_facecolor (optional): Facecolor of the figure. Default is 'white'.


        Returns:
            None

        Raises:
            None

        Usage:
            instance_name.plot_fisher(axis_custom_factors=create_factors(),
                                        filled=True,
                                        contour_args=[{'alpha':0.7}],
                                        legend_loc='lower left',
                                        dpi=150,
                                        file_format='.png',
                                        marker_color='red',
                                        axes_fontsize=16,
                                        legend_fontsize=18,
                                        figure_legend_frame=10,
                                        axes_labelsize=14)
        """
        self.load_gaussians()

        print("Entering plotting routine")

        # THIS MUST BE CHANGED
        # In principle the fiducials could be different!
        cust_lims = kwargs.get('axis_custom_factors', None)
        filled_ = kwargs.get('filled', True)
        contour_args_ = kwargs.get('contour_args', [{'alpha': 0.9}])
        legend_loc_ = kwargs.get('legend_loc', 'upper right')
        dpi_ = kwargs.get('dpi', 300)
        format_ = kwargs.get('file_format', '.pdf')
        marker_color_ = kwargs.get('marker_color', 'black')
        axes_fontsize = kwargs.get('axes_fontsize', 20)
        legend_fontsize = kwargs.get('legend_fontsize', 20)
        figure_legend_frame = kwargs.get('figure_legend_frame', 20)
        axes_labelsize = kwargs.get('axes_labelsize', 20)
        figure_facecolor = kwargs.get('figure_facecolor', "white")
        g = plots.get_subplot_plotter(
            subplot_size=1, width_inch=12, scaling=False)
        g.settings.figure_legend_frame = figure_legend_frame
        g.settings.axes_fontsize = axes_fontsize
        g.settings.axes_labelsize = axes_labelsize
        g.settings.legend_fontsize = legend_fontsize
        g.settings.axis_marker_color = marker_color_
        g.settings.axis_marker_ls = '--'
        g.settings.axis_marker_lw = 2
        g.triangle_plot(
            self.gaussians,
            self.plot_pars,
            filled=filled_,
            legend_labels=self.fish_labels,
            legend_loc=legend_loc_,
            contour_colors=self.colors,
            contour_args=contour_args_,
            markers=self.fiducial_markers,
            param_limits=self.param_limits_bounds(
                axis_custom_factors=cust_lims))
        g.fig.align_ylabels()
        g.fig.align_xlabels()
        g.fig.set_facecolor(figure_facecolor)
        if self.options['outroot'] is not None:
            contstr = self.options.get('contours_str', '_contours')
            g.fig.savefig(
                os.path.join(
                    self.options['outpath'],
                    self.options['outroot'] +
                    contstr +
                    format_),
                dpi=dpi_,
                bbox_inches='tight')

        return None

    def compare_errors(self, options=dict()):
        imgformat_ = options.get('file_format', '.pdf')
        plot_style = options.get('plot_style', 'bars')
        save_error = options.get('save_error', False)
        fishlabsjoin = ("-").join(self.fish_labels)
        fishlabsjoin.replace(" ", "_")
        ncol_legend = options.get('ncol_legend', None)
        legend_title = options.get('legend_title', None)
        legend_title_fontsize = options.get('legend_title_fontsize', None)
        # ffs.mkdirp(options['outpath'])
        errstr = options.get('errors_str', '_error_comparison')
        marginalze_remaining_pars = options.get(
            'marginalize_remaining_pars', True)
        plot_marg = options.get('plot_marg', True)
        plot_unmarg = options.get('plot_unmarg', True)
        xticksrotation = options.get('xticksrotation', 0)
        xticklabsize = options.get('xticklabsize', 22)
        yticklabsize = options.get('yticklabsize', 22)
        xtickfontsize = options.get('xtickfontsize', 22)
        ylabelfontsize = options.get('ylabelfontsize', 20)
        patches_legend_fontsize = options.get('patches_legend_fontsize', 26)
        dots_legend_fontsize = options.get('dots_legend_fontsize', 26)
        yrang = options.get('yrang', [-1., 1.])
        dpi = options.get('dpi', 400)
        figsize = options.get('figsize', (20, 10))
        transform_latex_dict = options.get('transform_latex_dict', dict())
        figure_title = options.get('figure_title', '')
        pc.ploterrs(
            self.fishers_group.get_fisher_list(),
            self.fish_labels,
            parstoplot=self.plot_pars,
            plot_style=plot_style,
            marginalize_pars=marginalze_remaining_pars,
            outpathfile=os.path.join(
                self.outpath,
                self.options['outroot'] + errstr + imgformat_),
            plot_marg=plot_marg,
            plot_unmarg=plot_unmarg,
            yrang=yrang,
            figsize=figsize,
            dpi=dpi,
            savefig=True,
            y_label=r'% discrepancy on $\sigma_i$ w.r.t. median',
            yticklabsize=yticklabsize,
            xticklabsize=xticklabsize,
            xtickfontsize=xtickfontsize,
            ylabelfontsize=ylabelfontsize,
            xticksrotation=xticksrotation,
            patches_legend_fontsize=patches_legend_fontsize,
            dots_legend_fontsize=dots_legend_fontsize,
            fish_leg_loc='lower right',
            legend_title=legend_title,
            legend_title_fontsize=legend_title_fontsize,
            ncol_legend=ncol_legend,
            transform_latex_dict=transform_latex_dict,
            save_error=save_error,
            figure_title=figure_title)

    def matrix_ratio(self, r_fishers_list=None,
                     tick_labels=None, plot_title=None, ratio_mat=None,
                     filename=None, savefig=True):
        imgformat_ = self.file_format
        if r_fishers_list is None:
            r_fishers_list = self.fishers_group.fisher_list[0:2]
        if tick_labels is None:
            tick_labels = [r'${}$'.format(ii) for ii in self.param_labels]
        if plot_title is None:
            plot_title = fishlabsjoin = ("/").join(self.fish_labels)
            plot_title = fishlabsjoin.replace(" ", "_")
            plot_title = 'Ratio ' + plot_title
        if ratio_mat is None:
            ratio_mat = (
                r_fishers_list[0].fisher_matrix /
                r_fishers_list[1].fisher_matrix)
        if filename is None:
            plot_name = plot_title.replace("/", "-")
            plot_name = plot_name.replace(" ", '_')
            matstr = self.options.get('matrix_str', '_matrix_ratio')
            filename = os.path.join(
                self.outpath,
                self.options['outroot'] +
                matstr +
                imgformat_)

        pc.matrix_plot(
            ratio_mat,
            xlabel=plot_title,
            ticklabels=tick_labels,
            filename=filename,
            figsize=(
                9,
                9),
            colormap=plt.cm.viridis,
            savefig=savefig,
            dpi=200)

import numpy as np

from cosmicfishpie.analysis import fisher_operations as cfo
from cosmicfishpie.analysis import fisher_plotting as cfp
from cosmicfishpie.utilities.utils import printing as cpr


def test_fisheroperations(spectro_fisher_matrix):
    cpr.debug = False
    fish = spectro_fisher_matrix.compute(max_z_bins=1)

    # Marginalisation
    post = cfo.marginalise_over(fish, ["h"])
    assert "h" not in post.get_param_names()
    assert np.isclose(fish.get_confidence_bounds()[0], post.get_confidence_bounds()[0])
    assert np.isclose(
        post.get_confidence_bounds(marginal=False)[0], 0.007891362317701697, rtol=1.0e-3
    )

    # Fixing
    post = cfo.eliminate_parameters(fish, ["h"])
    assert "h" not in post.get_param_names()
    assert np.isclose(
        fish.get_confidence_bounds(marginal=False)[0], post.get_confidence_bounds(marginal=False)[0]
    )
    assert np.isclose(post.get_confidence_bounds()[0], 0.00333069850284891, rtol=1.0e-3)

    # Reshuffeling
    post = cfo.reshuffle(fish, ["Omegam", "lnbg_1", "Ps_1"])
    assert np.isclose(post.get_confidence_bounds()[0], 0.00333069850284891, rtol=1.0e-3)


def test_getdist_plotters(spectro_fisher_matrix):
    cpr.debug = False
    fish = spectro_fisher_matrix.compute(max_z_bins=1)
    plotpars = ["Omegam", "h"]
    post = cfo.reshuffle(fish, plotpars)
    fisher_list = [fish, post]

    fish_plotter = cfp.fisher_plotting(
        fishers_list=fisher_list,
        plot_pars=plotpars,
        outroot=None,
        plot_method="Gaussian",
        colors=["blue", "red"],
    )
    fish_plotter.plot_fisher()
    fish_plotter.compare_errors(options={"savefig": False})

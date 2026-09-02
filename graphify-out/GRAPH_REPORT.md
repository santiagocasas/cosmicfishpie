# Graph Report - .  (2026-08-06)

## Corpus Check
- 160 files · ~105,410 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1689 nodes · 2486 edges · 170 communities (118 shown, 52 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 58 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Spectroscopic covariance calculations
- Cosmology backend interface
- Photometric benchmark tooling
- Configuration system tests
- Derived Fisher tests
- Core package modules
- Cosmological background functions
- Fisher analysis operations
- Spectroscopic likelihood calculations
- Comparison report generation
- Project documentation
- Fisher matrix computation
- Legendre quadrature tests
- Photometric angular spectra
- FishConsumer utility functions
- Fisher matrix accessors
- Numerical derivative calculations
- Spectroscopic power spectrum
- Derived parameter transformations
- Base likelihood tests
- Photometric covariance calculations
- Spectroscopic observable setup
- Spectroscopic observable tests
- FishConsumer plotting workflow
- Analysis package utilities
- Plot color utilities
- Terminal color formatting
- Fisher parameter operations
- Fisher matrix validation
- Terminal color tests
- Spectroscopic numerical stability tests
- CMB covariance calculations
- Planck best-fit Fisher
- Fisher matrix serialization
- Survey nuisance models
- FishConsumer utility tests
- Fisher analysis tests
- Spectroscopic array input tests
- Printing utility tests
- Likelihood base classes
- Fisher directory comparison
- Planck covariance comparison
- Fisher comparison plots
- Likelihood evaluation methods
- Spectroscopic validation tests
- Numerical utility tests
- Nautilus sampling workflow
- Photometric galaxy distributions
- Backend Fisher comparison
- Photometric likelihood calculations
- Nonlinear power spectrum
- CMB Fisher smoke test
- Spectroscopic configuration tests
- Fisher plotting interface
- Spectroscopic performance tests
- Nuisance redshift binning
- Numerical helper functions
- Published Planck comparison
- FishConsumer class tests
- Advanced Fisher analysis tests
- Filesystem utility tests
- Fisher information operations
- CMB angular spectra
- Fisher comparison plots
- Nuisance model tests
- Photometric observable tests
- Spectroscopic test fixtures
- Project overview documentation
- Scientific number formatting
- Boltzmann validation configurations
- Spectroscopic bias interpolation
- Legacy photometric likelihood
- Old photometric likelihood
- CAMB photometric likelihood
- FishConsumer color tests
- FishConsumer constants tests
- Photometric likelihood tests
- Dictionary merge tests
- Fisher uncertainty utilities
- Continuous integration setup
- Photometric covariance tests
- Scalar Fisher priors
- Default FishConsumer instance
- Euclid photometric specifications
- Euclid ISTF specifications
- Contribution workflow documentation
- Benchmark packaging script
- FishConsumer uncertainty tests
- FishConsumer LaTeX tests
- FishConsumer chain loading
- Physical constants tests
- INI parser tests
- CLASS solver configurations
- Euclid spectroscopic specifications
- MeerKAT intensity mapping
- SKAO intensity mapping
- SKAO spectroscopic specifications
- Backend comparison report
- CMB benchmark runner
- Fisher matrix equality
- External data installation
- Euclid q-bias specifications
- Euclid q-bias sigma-pv
- Euclid sigma-pv specifications
- SKAO photometric specifications
- Sampler chain metadata
- Transverse Alcock-Paczynski scaling
- Project changelog documentation
- Sphinx configuration
- Contribution documentation
- Dark energy sampler configurations
- Release notes generation
- MontePython chain loading
- Euclid one-parameter q-bias
- INI configuration parser
- Planck diagnostics runner
- Comparison configuration validator
- Quality check configuration
- Backend performance guidance
- FishConsumer Fisher preparation
- Fisher LaTeX parameter names
- Fisher fiducial parameters
- Fisher LaTeX name setter
- Release script
- Terminal color constants test
- Terminal header color test
- Terminal color initialization test
- Named Fisher retrieval test
- All Fisher retrieval test
- Fisher deletion test
- Fisher parameter list test
- Fisher LaTeX names test
- Fisher reshuffling test
- Fisher marginalization test
- Fisher comparison test
- Fisher plot range test
- Fisher Gaussian test
- Fisher ellipse test
- Multiple Fisher operations test
- Fisher error handling test
- Fisher list initialization test
- Semantic versioning policy
- Default CAMB configuration
- CAMB HMCode configuration
- Default CLASS configuration
- Fast photometric CLASS configuration
- Default symbolic configuration
- Symbolic Halofit configuration
- Symbolic Syren configuration
- CMB Stage-4 configuration
- Euclid photometric default configuration
- Euclid spectroscopic default configuration
- Planck survey configuration
- Simons Observatory configuration
- Dependency update automation
- Bug report template
- Documentation issue template
- Feature request template
- Package root module
- Scientific Python dependencies

## God Nodes (most connected - your core abstractions)
1. `fisher_matrix` - 54 edges
2. `ComputeGalSpectro` - 50 edges
3. `FisherMatrix` - 41 edges
4. `printing` - 40 edges
5. `Nuisance` - 32 edges
6. `TestCosmicFishFisherAnalysis` - 27 edges
7. `FishConsumer` - 26 edges
8. `bash_colors` - 25 edges
9. `SpectroCov` - 23 edges
10. `derivatives` - 22 edges

## Surprising Connections (you probably didn't know these)
- `TestArrayHandling` --uses--> `ComputeGalSpectro`  [INFERRED]
  tests/reference_spectro_obs_edge_cases.py → cosmicfishpie/LSSsurvey/spectro_obs.py
- `TestConfigurationOptions` --uses--> `ComputeGalSpectro`  [INFERRED]
  tests/reference_spectro_obs_edge_cases.py → cosmicfishpie/LSSsurvey/spectro_obs.py
- `TestErrorHandlingAndValidation` --uses--> `ComputeGalSpectro`  [INFERRED]
  tests/reference_spectro_obs_edge_cases.py → cosmicfishpie/LSSsurvey/spectro_obs.py
- `TestNumericalStability` --uses--> `ComputeGalSpectro`  [INFERRED]
  tests/reference_spectro_obs_edge_cases.py → cosmicfishpie/LSSsurvey/spectro_obs.py
- `TestPerformanceAndScaling` --uses--> `ComputeGalSpectro`  [INFERRED]
  tests/reference_spectro_obs_edge_cases.py → cosmicfishpie/LSSsurvey/spectro_obs.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **CLASS Solver Configuration Profiles** — boltzmann_yaml_files_class_default_class_default_settings, boltzmann_yaml_files_class_fast_photo_fast_photometric_settings, boltzmann_yaml_files_class_fast_photo_hmcode_fast_photometric_hmcode_settings, boltzmann_yaml_files_class_fast_spectro_fast_spectroscopic_settings [INFERRED 0.95]
- **Documented Cosmicfishpie Packages** — docs_source_cosmicfishpie_cmbsurvey_cmbsurvey_package, docs_source_cosmicfishpie_lsssurvey_lsssurvey_package, docs_source_cosmicfishpie_analysis_analysis_package, docs_source_cosmicfishpie_cosmology_cosmology_package, docs_source_cosmicfishpie_fishermatrix_fishermatrix_package, docs_source_cosmicfishpie_utilities_utilities_package [EXTRACTED 1.00]
- **Dark Energy Sampler Configurations** — sampler_scripts_darkenergy_desi_config_desi_w0wa_fiducial, sampler_scripts_darkenergy_lcdm_config_lcdm_fiducial, sampler_scripts_darkenergy_lcdm_highh0_config_lcdm_high_h0_fiducial [INFERRED 0.85]

## Communities (170 total, 52 thin omitted)

### Community 0 - "Spectroscopic covariance calculations"
Cohesion: 0.05
Nodes (27): Calculates Tsys in mK Parameters ---------- z : float, numpy.ndarray Redshift…, Calculates the comoving volume of a spherical shell Parameters ---------- zi :…, Calculates the comoving volume of a redshift bin Parameters ---------- i : int…, Calculates the survey volume of a redshift bin Parameters ---------- i : int…, calculate the comoving number density of the probe Parameters ---------- i :…, calculate the effective volume entering the covariance of the galaxy clustering…, Initializes an object with specified fiducial parameters and computes various…, Function to calculate the covariance the galaxy clustering probe Parameters… (+19 more)

### Community 1 - "Cosmology backend interface"
Cohesion: 0.06
Nodes (27): boltzmann_code, _dcom_func_trapz(), external_input, memorize_external_input(), Set up parameters for symbolic computation. Notes ----- This method prepares…, Compute and store results using symbolic computation. Returns -------…, Initialize the external_input class. Parameters ---------- cosmopars : dict The…, Load text files containing cosmological data. Parameters ----------… (+19 more)

### Community 2 - "Photometric benchmark tooling"
Cohesion: 0.06
Nodes (40): faster_integral_efficiency(), memo_integral_efficiency(), much_faster_integral_efficiency(), O(N) algorithm for the lensing efficiency integral using backward cumulative…, Legacy O(N^2) implementation of the efficiency integral. Notes ----- -…, computes the integral that enters the lensing kernel for a given redshift bin…, function to do the integration over redshift that shows up in the lensing…, load_fisher_from_json() (+32 more)

### Community 3 - "Configuration system tests"
Cohesion: 0.05
Nodes (24): Test suite for cosmicfishpie.configs.config module. This module tests…, Test config.init() with different survey names., Test config.init() with different cosmological models., Test configuration constants and module structure., Test that the config module has expected attributes., Test function signatures and docstrings., Test that imported modules are accessible., Test file operations and path handling. (+16 more)

### Community 4 - "Derived Fisher tests"
Cohesion: 0.05
Nodes (20): Test suite for cosmicfishpie.analysis.fisher_derived module. This module tests…, Test get_derived_param_fiducial method., Test initialization with all parameters., Test load_paramnames_from_file method., Test the fisher_derived class., Test initialization from file., Test add_derived method., Test basic fisher_derived initialization. (+12 more)

### Community 5 - "Core package modules"
Cohesion: 0.13
Nodes (12): :synopsis: Module that contains the fisher_plotting class and related functions…, # TODO: fix pandas stuff, # TODO: fix pandas stuff!!, # TODO: Fix for non-flat models, # TODO: nonlinear options to be selectable, # TODO: can be optimized by returning interpolating function in z and, filesystem, misc (+4 more)

### Community 6 - "Cosmological background functions"
Cohesion: 0.07
Nodes (19): init(), This class is to handle the configuration for the fishermatrix computation as…, cosmo_functions, Hubble function Parameters ---------- z : float redshift physical: bool Default…, E(z) dimensionless Hubble function Parameters ---------- z : float redshift…, Angular diameter distance Parameters ---------- z : float redshift Returns…, Calculates the power spectrum of a given tracer quantity at a specific redshift…, Compute the power spectrum of the total matter species (MM) at a given redshift… (+11 more)

### Community 7 - "Fisher analysis operations"
Cohesion: 0.09
Nodes (18): CosmicFish_FisherAnalysis, CosmicFish_FisherAnalysis class destructor. Makes sure everything is gone when…, Searches a path for fisher matrices. Will detect wether fisher_path contains…, Add a set of Fisher matrices to the already existing set. Rejects Fisher…, Returns the list of Fisher matrices corresponding to the given names. :param…, Delete the fisher matrix or the fisher matrices in names from the Fisher list.…, Returns the list of parameter names of all the matrices identified in names.…, Reshuffles all the Fisher matrices. :param params: parameters to reshuffle.… (+10 more)

### Community 8 - "Spectroscopic likelihood calculations"
Cohesion: 0.12
Nodes (24): compute_chi2_legendre(), compute_covariance_legendre(), compute_theory_spectro(), compute_wedge_chi2(), _dict_with_updates(), legendre_Pgg(), loglike(), observable_Pgg() (+16 more)

### Community 9 - "Comparison report generation"
Cohesion: 0.19
Nodes (33): bundle_reports(), _bundle_run_folder(), _copy_file(), _copy_tree(), _data_uri(), _discover_run_folders(), _find_compare_json(), _find_latest_reported_pair_files() (+25 more)

### Community 10 - "Project documentation"
Cohesion: 0.07
Nodes (32): CMB Benchmark Presets, CMB Fisher Path, CMB T E B Observables, CMB Fisher Smoke Run, CMB Survey Specifications, Analysis Package, Fisher Analysis Modules, CMB Covariance Module (+24 more)

### Community 11 - "Fisher matrix computation"
Cohesion: 0.09
Nodes (16): FisherMatrix, This will print all the selected options into the standard output, This function will compute the Fisher information matrix and export using the…, Function to define grids of the internal wavenumber and observation angle Note…, This computes the derivatives of the observed power spectrum for all redshift…, This computes the Fisher matrix of a spectroscopic probe for all redshift bins…, This helper function contains the Fisher matrix of a spectroscopic probe for a…, This helper function calculates a singular element of the fisher matrix… (+8 more)

### Community 12 - "Legendre quadrature tests"
Cohesion: 0.08
Nodes (17): gauss_lobatto_abscissa_and_weights(), Test suite for cosmicfishpie.utilities.legendre_tools module. This module tests…, Test properties of m22 matrix (l3=l4=2)., Test properties of m44 matrix (l3=l4=4)., Test that matrices contain expected numerical ranges., Test that the original lists convert correctly to matrices., Test typical access patterns for the matrices., Test Gauss-Lobatto quadrature functions. (+9 more)

### Community 13 - "Photometric angular spectra"
Cohesion: 0.11
Nodes (14): ComputeCls, Main class to obtain the angular power spectrum of the photometric probe.…, Main function to compute the angular power spectrum. Will first compute the…, prints the numerical specifications of the internal computations, Calculates the Limber-approximated power spectrum. This is done for a range of…, Calculates the square root of the Limber-approximated power spectrum for weak…, Calculates the GCph kernel function Parameters ---------- z : numpy.ndarray…, Computes the photometric cosmic shear kernel function Parameters ---------- z :… (+6 more)

### Community 14 - "FishConsumer utility functions"
Cohesion: 0.08
Nodes (16): clamp(), display_colors(), fishtable_to_pandas(), hex2rgb(), load_Nautilus_chains_from_txt(), parse_log_param(), plot_chain_summary(), Create a summary plot of MCMC chains using ChainConsumer. This function… (+8 more)

### Community 15 - "Fisher matrix accessors"
Cohesion: 0.08
Nodes (14): fisher_matrix, Returns the number of a parameter as specified by his name. Notice this differs…, This class contains the relevant code to define a fisher matrix and basic…, Addition operator (+). Safeguarded agains adding Fisher matrices with different…, This function returns the determinant of the Fisher matrix. :return: a…, :returns: the eigenvalues of the Fisher matrix as a numpy array., :returns: the eigenvectors of the Fisher matrix., Function sets a new list of param names substituting the old one. Notice that… (+6 more)

### Community 16 - "Numerical derivative calculations"
Cohesion: 0.13
Nodes (16): derivatives, One of the possible derivative methods. Computes the numerical derivative using…, Helper function to compute the 4PT forward finite step size derivative…, r"""One of the possible derivative methods. Computes the numerical derivative…, This class is the main derivative engine for the different observables. It…, One of the possible derivative methods. Computes the numerical derivative using…, One of the possible derivative methods. Computes the numerical derivative using…, Helper function to compute the 3PT symmetrical finite step size derivative… (+8 more)

### Community 17 - "Spectroscopic power spectrum"
Cohesion: 0.11
Nodes (11): Function implementing q parallel of the Alcock-Paczynski effect Parameters…, Computes the parallel projection of a wavevector. Takes into acount AP-effect…, Function that rescales the k-array, when asked for. The code is defined…, Function rescaling k and mu with the Alcock-Paczynski effect Parameters…, Function to compute the scale dependant suppression of the observed power…, Calculates the BAO term. This is the rescaling of the Fourier volume by the AP-…, Calculate the fiducial bias term for galaxies or intensity mapping. Parameters…, Computes the Kaiser redshift-space distortion term. Parameters ---------- z :… (+3 more)

### Community 18 - "Derived parameter transformations"
Cohesion: 0.10
Nodes (12): fisher_derived, Loads the paramnames array, of a derived Fisher matrix, from a file :param…, This class contains the relevant code to define a matrix that contains the…, This function computes the derived fisher_matrix given an input Fisher matrix…, :returns: the derived Jacobian matrix., :returns: the base parameter names., :returns: the LaTeX version of the base parameter names., :returns: the base parameter fiducial values. (+4 more)

### Community 19 - "Base likelihood tests"
Cohesion: 0.09
Nodes (12): Test suite for cosmicfishpie.likelihood.base module. This module tests the base…, Test the base likelihood functionality., Test utility functions in the module., Test parameter validation functions., Test base likelihood initialization., Test that likelihood functions are accessible., Test basic likelihood computation if function exists., Test log likelihood computation if function exists. (+4 more)

### Community 20 - "Photometric covariance calculations"
Cohesion: 0.13
Nodes (14): PhotoCov, Function to calculate the angular power spectrum Parameters ---------- allpars…, Obtain the angular power spectrum with noise Parameters ---------- cls : dict a…, Computes the covariance matrix from the noisy angular power spectrum Parameters…, Computes the fiducial covariance matrix for the Fisher matrix. Returns -------…, Main class to obtain the ingredients for the Fisher matrix of a photometric…, computes the derivatives of the angular power spectrum needed to compute the…, computecls_fid() (+6 more)

### Community 21 - "Spectroscopic observable setup"
Cohesion: 0.14
Nodes (8): ComputeGalSpectro, Updates the internal grid of wavenumbers used in the computation, Update which modelling effects should be taken into consideration, Updates the spectroscopic redshift error, Updates the spectroscopic bias choices, Class to compute the observed power spectrum for spectroscopic galaxy…, Updates the IM bias choices, obtaining the temperature (T^2(z)) for the Power Spectrum (PHI(z))

### Community 23 - "FishConsumer plotting workflow"
Cohesion: 0.11
Nodes (3): choose_fish_toplot(), FishConsumer, Convenience wrapper that exposes the fishconsumer helpers as instance methods.

### Community 24 - "Analysis package utilities"
Cohesion: 0.12
Nodes (15): confidence_coefficient(), CosmicFish_write_header(), find_nearest(), grouper(), make_list(), mkdirp(), print_table(), This function returns the number of sigmas given a confidence level. See page… (+7 more)

### Community 25 - "Plot color utilities"
Cohesion: 0.15
Nodes (11): nice_colors(), This function returns a color from a colormap defined below according to the…, Test suite for cosmicfishpie.analysis.colors module. This module tests color…, Test the nice_colors function., Test nice_colors with integer inputs., Test nice_colors with float inputs., Test that nice_colors uses modulo 7., Test nice_colors with negative inputs. (+3 more)

### Community 26 - "Terminal color formatting"
Cohesion: 0.11
Nodes (10): bash_colors, Function that returns a string that can be printed to bash in…, Function that returns a string that can be printed to bash in…, Function that returns a string that can be printed to bash in…, Function that returns a string that can be printed to bash in…, Function that returns a string that can be printed to bash in…, Function that returns a string that can be printed to bash in…, This class contains the necessary definitions to print to bash screen with… (+2 more)

### Community 27 - "Fisher parameter operations"
Cohesion: 0.16
Nodes (15): eliminate_columns_rows(), eliminate_parameters(), marginalise(), marginalise_over(), This function marginalises a Fisher matrix over all parameters but the ones in…, This function marginalises a Fisher matrix over the parameters in names. The…, This function eliminates the row and columns corresponding to the given indexes…, This function eliminates the row and columns corresponding to the given… (+7 more)

### Community 28 - "Fisher matrix validation"
Cohesion: 0.13
Nodes (9): **fisher_matrix class constructor**. The constructor will read from file the…, Loads the paramnames array from a file. :param file_name: (optional) file name…, Invert the Fisher matrix. :returns: a matrix containing the inverse of the…, Assert if the Fisher matrix is symmetric or not :returns: a :class:`bool`…, Transforms a non-symmetric matrix into a symmetric matrix, This function performs the principal component analysis of the Fisher matrix…, Protects the Fisher matrix against degeneracies. Modifies the spectrum to…, Computes the marginal 1D confidence bounds on the Fisher parameters :param… (+1 more)

### Community 29 - "Terminal color tests"
Cohesion: 0.11
Nodes (10): Test the blue method., Test the green method., Test the warning method., Test the bold method., Test the underline method., Test all color methods with numeric inputs., Test all color methods with empty string., Test the bash_colors class. (+2 more)

### Community 30 - "Spectroscopic numerical stability tests"
Cohesion: 0.11
Nodes (10): Test behavior with large spectroscopic error., Test monotonicity properties where expected., Test AP effect in limiting cases., Test consistency of nonlinear terms., Test numerical stability and edge cases., Test behavior with very small input values., Test behavior with very large input values., Test extreme mu values. (+2 more)

### Community 31 - "CMB covariance calculations"
Cohesion: 0.15
Nodes (9): CMBCov, Combine multiple channels by inverse-variance weighting., Combine channels as inverse-noise weighted sum., Data covariance Parameters ---------- noisy_cls : dict dictionary containing…, CMB covariance and derivatives for Fisher forecasts. This class: - computes…, Compute fiducial CMB spectra, add noise, and build covariance matrices. Returns…, Compute numerical derivatives of CMB spectra w.r.t. free parameters., Compute (noise-free) CMB C_ell for a given parameter dictionary. (+1 more)

### Community 32 - "Planck best-fit Fisher"
Cohesion: 0.29
Nodes (16): _build_planck_camb_yaml(), _get_h_from_bestfit(), _git_commit(), main(), _parse_inputparams(), _parse_likestats(), _parse_margestats(), _parse_minimum() (+8 more)

### Community 33 - "Fisher matrix serialization"
Cohesion: 0.16
Nodes (8): Returns the name of the parameter corresponding to the given number. :param…, Returns the index of a parameter as specified by his name. Notice that indices…, Returns the Latex name of the parameter called name. :param name: input name or…, Returns the fiducial of the parameter called name. :param name: input name or…, Saves the paramnames to a file. :param file_name: (optional) file name and path…, Saves the fisher matrix to a file. Notice that the file name has to be…, :returns: the fisher matrix as a numpy array., :returns: the fiducial values of the parameters of the Fisher matrix.

### Community 34 - "Survey nuisance models"
Cohesion: 0.13
Nodes (6): Nuisance, r"""Intrinsic Alignment :param z: float redshift :return: - float: Value of IA…, IM 21cm HI bias function from http://arxiv.org/abs/2006.05996, Create interpolation function for HI intensity mapping system noise…, Galaxy bias Parameters ---------- z : array redshift Returns ------- float…, test_bterm_fid()

### Community 35 - "FishConsumer utility tests"
Cohesion: 0.12
Nodes (9): Test arrays_gaussian function., Test utility functions in fishconsumer module., Test the clamp function., Test RGB to hex conversion., Test LaTeX formatting function., Test percentage to absolute conversion., Test log fiducial to fiducial conversion., Test Gaussian function. (+1 more)

### Community 36 - "Fisher analysis tests"
Cohesion: 0.12
Nodes (9): Test search_fisher_path method., Test operations on empty analysis object., Test file-related operations with mocking., Test the CosmicFish_FisherAnalysis class., Test basic CosmicFish_FisherAnalysis initialization., Test get_fisher_list method., Test get_fisher_name_list method., Test initialization with fisher_path parameter. (+1 more)

### Community 37 - "Spectroscopic array input tests"
Cohesion: 0.12
Nodes (9): Test array input handling and broadcasting., Test with all scalar inputs., Test with array of redshifts., Test with array of wavenumbers., Test with array of mu values., Test with mixed array and scalar inputs., Test with 2D arrays (if supported)., Test with empty arrays. (+1 more)

### Community 38 - "Printing utility tests"
Cohesion: 0.12
Nodes (9): Test time_print respects feedback level filtering., Test suppress_warnings decorator when enabled., Test suppress_warnings decorator when disabled., Test the printing utility class., Test debug_print when debug is enabled., Test debug_print when debug is disabled., Test time_print with time measurements., Test time_print with instance parameter. (+1 more)

### Community 39 - "Likelihood base classes"
Cohesion: 0.19
Nodes (10): ABC, Likelihood, NautilusMixin, Base infrastructure for likelihood modules in Cosmicfishpie. This module…, Mixin class for running Nautilus samplers., Common interface for likelihood evaluations used in Cosmicfishpie. This…, Initialize the Likelihood object with Fisher matrices. Args: cosmo_data:…, Compute and return the observed data representation. This method should be… (+2 more)

### Community 40 - "Fisher directory comparison"
Cohesion: 0.26
Nodes (14): _align_matrices(), _analysis_pair(), _load_specs(), _load_yaml(), main(), _metrics(), ndarray, Path (+6 more)

### Community 41 - "Planck covariance comparison"
Cohesion: 0.30
Nodes (13): make_triangle_plot(), Create a triangle plot from Fisher matrices and/or MCMC chains using…, _corr_from_cov(), main(), _parse_likestats_bestfit(), _parse_paramnames(), ndarray, Path (+5 more)

### Community 42 - "Fisher comparison plots"
Cohesion: 0.21
Nodes (13): add_colorbar(), calc_y_range(), matrix_plot(), og_plot_shades(), plot_shades(), ploterrs(), process_fish_errs(), :synopsis: Module for creating comparison plots of Fisher Matrix entries and… (+5 more)

### Community 43 - "Likelihood evaluation methods"
Cohesion: 0.19
Nodes (9): is_indexable_iterable(), Any, Create a parameter dictionary from vector inputs when needed. This method…, Compute the log-likelihood value. This method computes the log-likelihood value…, Convenience wrapper to launch a Nautilus sampler using this likelihood. This…, Check if a variable is an indexable iterable. Args: var: The variable to check…, Compute and return the theory prediction. This method should be implemented by…, Compute and return the chi-squared value. This method should be implemented by… (+1 more)

### Community 44 - "Spectroscopic validation tests"
Cohesion: 0.14
Nodes (8): Test error handling and input validation., Create a ComputeGalSpectro instance for testing., Test error handling for invalid bias samples., Test handling of NaN inputs., Test handling of infinite inputs., Test handling of negative inputs where inappropriate., Test handling of mu values outside [-1, 1]., TestErrorHandlingAndValidation

### Community 45 - "Numerical utility tests"
Cohesion: 0.14
Nodes (8): Test the numerics utility class., Test basic moving average calculation., Test moving average with different periods., Test round_decimals_up method., Test closest function., Test bisection search function., Test find_nearest function., TestNumericsClass

### Community 46 - "Nautilus sampling workflow"
Cohesion: 0.24
Nodes (3): Create a Nautilus prior object from a dictionary of parameter names and their…, NautilusSampler, Prior

### Community 47 - "Photometric galaxy distributions"
Cohesion: 0.22
Nodes (7): GalaxyPhotoDist, Function to compute the binned galaxy redshift distribution convolved with…, Class to obtain the survey specific ingredients of the window function…, n^{ph}_i(z) Parameters ---------- z : float redshift at which to compute the…, n^{ph}_i(z) Parameters ---------- z : array redshift at which to compute the…, unnormalized dN/dz(z) Parameters ---------- z : numpy.ndarray array of…, Function to compute the unnormalized dN/dz(z) with a window picking function…

### Community 48 - "Backend Fisher comparison"
Cohesion: 0.35
Nodes (12): Namespace, _build_base_options(), _default_paths(), _git_commit(), _infer_yaml_key(), _load_common_specs(), main(), _make_run_id() (+4 more)

### Community 49 - "Photometric likelihood calculations"
Cohesion: 0.30
Nodes (8): _cells_from_cls(), _chi2_per_obs(), _dict_with_updates(), PhotometricLikelihood, Any, ComputeCls, ndarray, Likelihood built from photometric clusterings (WL / GCph).

### Community 50 - "Nonlinear power spectrum"
Cohesion: 0.17
Nodes (6): Function to calculate the variance of the velocity dispersion Parameters…, Function to calculate the variance of the displacement field Parameters…, Calculates the angular power spectrum moments of the velocity divergence field,…, This function normalizes the power spectrum to have a variance smoothed over 8…, This function normalizes the power spectrum with the BAO wiggles subtracted…, This function calculates the normalized dewiggled power spectrum. Parameters…

### Community 51 - "CMB Fisher smoke test"
Cohesion: 0.41
Nodes (11): _enable_cmb_in_boltzmann_yaml(), _git_commit(), _infer_yaml_key(), _load_yaml(), main(), Any, Path, _repo_root() (+3 more)

### Community 52 - "Spectroscopic configuration tests"
Cohesion: 0.17
Nodes (7): Test different configuration options and switches., Test linear vs nonlinear modeling switch., Test Fingers of God switch., Test Alcock-Paczynski effect switch., Test different spectroscopic error types., Test h-rescaling bug flag behavior., TestConfigurationOptions

### Community 53 - "Fisher plotting interface"
Cohesion: 0.24
Nodes (3): fisher_plotting, Generates a triangle plot based on loaded gaussian data and specified…, This class uses the cosmicfish_pylib classes to generate contour plots using…

### Community 54 - "Spectroscopic performance tests"
Cohesion: 0.18
Nodes (6): Test performance characteristics and scaling., Test performance of single evaluation., Test performance with array inputs., Test repeated evaluations for potential caching issues., Test that repeated evaluations don't cause memory leaks., TestPerformanceAndScaling

### Community 55 - "Nuisance redshift binning"
Cohesion: 0.20
Nodes (4): Luminosity ratio function used for Intrinsic Alignment eNLA model. This…, Reads from file for a given survey, Reads from file for a given survey, Reads from file for a given survey

### Community 56 - "Numerical helper functions"
Cohesion: 0.20
Nodes (4): numerics, Returns a value rounded up to a specific number of decimal places., Given an ``array``, and given a ``value``, returns an index j such that…, Element in nd array `a` closest to the scalar value `a0`

### Community 57 - "Published Planck comparison"
Cohesion: 0.44
Nodes (9): _collect_fisher_sigmas(), main(), _parse_margestats(), _print_table(), Path, _repo_root(), _resolve_fisher_path(), _to_h_from_h0() (+1 more)

### Community 58 - "FishConsumer class tests"
Cohesion: 0.20
Nodes (6): Test the FishConsumer wrapper class., Test FishConsumer initialization., Test FishConsumer initialization with custom colors., Test that methods are properly delegated to module functions., Test percentage to absolute conversion method., TestFishConsumerClass

### Community 59 - "Advanced Fisher analysis tests"
Cohesion: 0.20
Nodes (6): make_fisher(), Test the ``__del__`` method clears internal lists., Create a minimal positive definite ``fisher_matrix`` instance. Parameters…, Test union of parameter names across distinct matrices., Test operations on richer matrix (5 parameters)., Test ``add_fisher_matrix`` with a real fisher matrix.

### Community 60 - "Filesystem utility tests"
Cohesion: 0.20
Nodes (6): Test the filesystem utility class., Test mkdirp creates parent directory., Test mkdirp with existing directory., Test mkdirp creates nested parent directories., Test git_version function., TestFilesystemClass

### Community 61 - "Fisher information operations"
Cohesion: 0.22
Nodes (9): information_gain(), This function reshuffles a Fisher matrix. The new Fisher matrix will have the…, This function computes the Fisher approximation of Kullback-Leibler information…, reshuffle(), test_getdist_plotters(), Exercise information_gain for both stat=False and stat=True branches. The…, test_information_gain_stat_paths(), test_reshuffle_missing_param() (+1 more)

### Community 62 - "CMB angular spectra"
Cohesion: 0.28
Nodes (5): ComputeCls, Print the contents of `cfg.specs` (debug helper)., Return a dictionary of CMB C_ell arrays. Returns ------- dict Dictionary with…, Compute CMB angular power spectra for the configured observables. Parameters…, Compute all requested CMB spectra and store them in `self.result`.

### Community 63 - "Fisher comparison plots"
Cohesion: 0.58
Nodes (8): _code_labels(), _load(), main(), _pair_id(), _plot_fom(), _plot_param_differences(), Path, _safe_name()

### Community 64 - "Nuisance model tests"
Cohesion: 0.42
Nodes (8): _base_specs(), _make_config(), test_gcph_bias_binned_model_clips_out_of_range(), test_gcsp_rescale_sigmapv_default_and_named_key(), test_gcsp_zvalue_to_zindex_clamps_bounds(), test_ia_enla_returns_callable_spline(), test_im_bias_fitting_and_thi_noise_interp(), test_luminosity_ratio_missing_file_returns_unity()

### Community 65 - "Photometric observable tests"
Cohesion: 0.22
Nodes (6): Check that enabling/disabling _USE_FAST_KERNEL yields the same WL kernel. We…, Check that enabling/disabling _USE_FAST_EFF yields the same lensing efficiency., Check that enabling/disabling _USE_FAST_P yields identical Pell., test_P_limber_fast_P_equivalence(), test_wl_efficiency_fast_eff_equivalence(), test_wl_kernel_fast_kernel_equivalence()

### Community 66 - "Spectroscopic test fixtures"
Cohesion: 0.22
Nodes (5): fixture, Create a ComputeGalSpectro instance for testing., Create a ComputeGalSpectro instance for testing., Create a ComputeGalSpectro instance for testing., Create a ComputeGalSpectro instance for testing.

### Community 67 - "Project overview documentation"
Cohesion: 0.25
Nodes (8): CAMB Default Settings, CAMB, cosmicfishpie, Cosmological Fisher Forecasts, Cosmology Backends, FisherMatrix.compute, Post-processing Tools, Survey Specifications

### Community 68 - "Scientific number formatting"
Cohesion: 0.29
Nodes (8): mant_exp_to_num(), nice_number(), num_to_mant_exp(), This function returns the number in num_err at the precision of error. :param…, This function returns the (base 10) exponent and mantissa of a number. :param…, This function returns a float built with the given (base 10) mantissa and…, This function returns a nice number built with num. This is useful to build the…, significant_digits()

### Community 69 - "Boltzmann validation configurations"
Cohesion: 0.25
Nodes (8): CAMB Matter-Power Validation Configuration, CAMB Neutrino Validation Configuration, CLASS Matter-Power Validation Configuration, CLASS Photometric Neutrino Validation Configuration, CLASS Spectroscopic Neutrino Validation Configuration, Archidiacono et al. 2405.06047, Boltzmann Solver Configuration Folder, Casas et al. 2303.09451

### Community 70 - "Spectroscopic bias interpolation"
Cohesion: 0.25
Nodes (3): Parameters ---------- zi : int Redshift bin index Returns ------- float Bias at…, Parameters ---------- z : float Redshift Returns ------- float Bias at the…, Galaxy bias for the galaxies used in spectroscopic Galaxy Clustering Parameters…

### Community 71 - "Legacy photometric likelihood"
Cohesion: 0.43
Nodes (7): compute_chi2(), compute_chi2_per_obs(), is_indexable_iterable(), loglike(), observable_Cell(), ComputeCls, Compute χ² for wedges using fully vectorized operations. Matches the loop…

### Community 72 - "Old photometric likelihood"
Cohesion: 0.43
Nodes (7): compute_chi2(), compute_chi2_per_obs(), is_indexable_iterable(), loglike(), observable_Cell(), ComputeCls, Compute χ² for wedges using fully vectorized operations. Matches the loop…

### Community 73 - "CAMB photometric likelihood"
Cohesion: 0.43
Nodes (7): compute_chi2(), compute_chi2_per_obs(), is_indexable_iterable(), loglike(), observable_Cell(), ComputeCls, Compute χ² for wedges using fully vectorized operations. Matches the loop…

### Community 74 - "FishConsumer color tests"
Cohesion: 0.25
Nodes (5): Test color-related functions., Test that color constants are properly structured., Test color dictionary., Test display_colors function., TestColorFunctions

### Community 75 - "FishConsumer constants tests"
Cohesion: 0.25
Nodes (5): Test module constants and data structures., Test barplot filter names., Test parameter LaTeX names., Test LaTeX replacement dictionary., TestConstants

### Community 76 - "Photometric likelihood tests"
Cohesion: 0.36
Nodes (6): photometric_fiducial_obs(), photometric_likelihood(), fixture, _sample_params(), test_photometric_cell_entry_matches_theory(), test_photometric_loglike_matches_notebook_value()

### Community 77 - "Dictionary merge tests"
Cohesion: 0.25
Nodes (5): Test the misc utility class., Test basic deepupdate functionality., Test deepupdate when original is not a mapping., Test deepupdate with deeply nested dictionaries., TestMiscClass

### Community 78 - "Fisher uncertainty utilities"
Cohesion: 0.29
Nodes (3): arrays_gaussian(), n_sigmas(), sigma_fidu()

### Community 79 - "Continuous integration setup"
Cohesion: 0.29
Nodes (7): Setup Virtual Environment Action, uv Dependency Synchronization, CI Workflow, Main Workflow Checks, Release Workflow, GitHub Release Process, Release Script

### Community 80 - "Photometric covariance tests"
Cohesion: 0.29
Nodes (4): parametrize, Stub derivative engine so we only test integration & shape, not heavy…, test_photo_cov_cls_and_noise(), test_photo_cov_compute_derivs_stub()

### Community 81 - "Scalar Fisher priors"
Cohesion: 0.52
Nodes (6): main(), _overlap_params(), _print_compare_table(), Path, _resolve_fisher(), _sigma_map()

### Community 82 - "Default FishConsumer instance"
Cohesion: 0.29
Nodes (4): Test the default FishConsumer instance., Test that DEFAULT_FISH_CONSUMER exists and is properly initialized., Test that default instance methods work., TestDefaultInstance

### Community 83 - "Euclid photometric specifications"
Cohesion: 0.33
Nodes (6): Binned Photometric Bias Model, eNLA Intrinsic Alignment Model, Galaxy Clustering Photometric Configuration, Photometric Redshift Model, Euclid Photometric Debole R1 Survey Profile, Weak Lensing Configuration

### Community 84 - "Euclid ISTF specifications"
Cohesion: 0.33
Nodes (6): eNLA Intrinsic Alignment Model, Galaxy Clustering Photometric Configuration, Photometric Redshift Model, Square-Root Photometric Bias Model, Euclid Photometric ISTF GW Survey Profile, Weak Lensing Configuration

### Community 85 - "Contribution workflow documentation"
Cohesion: 0.33
Nodes (6): Local Quality Checks, Pull Request Workflow, Sphinx Docstrings, Pull Request Submission Checklist, Pull Request Changelog Check, Read the Docs Documentation Build

### Community 86 - "Benchmark packaging script"
Cohesion: 0.60
Nodes (5): _collect_files(), _git_cmd(), _git_meta(), main(), Path

### Community 87 - "FishConsumer uncertainty tests"
Cohesion: 0.33
Nodes (4): Test data processing functions., Test n_sigmas function., Test sigma_fidu function., TestDataProcessingFunctions

### Community 88 - "FishConsumer LaTeX tests"
Cohesion: 0.33
Nodes (4): Test functions that work with fisher matrices using mocks., Test replace_latex_name function with mock fisher matrix., Test replace_latex_style function with mock fisher matrix., TestMockFisherMatrix

### Community 89 - "FishConsumer chain loading"
Cohesion: 0.33
Nodes (4): Test functions that work with pandas DataFrames., Test loading Nautilus chains with mock data., Test loading Nautilus chains with log weights., TestPandasIntegration

### Community 91 - "Physical constants tests"
Cohesion: 0.33
Nodes (4): Test suite for cosmicfishpie.utilities.utils module. This module tests utility…, Test the physmath utility class., Test physmath class constants and attributes., TestPhysmathClass

### Community 92 - "INI parser tests"
Cohesion: 0.33
Nodes (4): Test the inputiniparser class., Test inputiniparser initialization., Test free_epsilons method., TestInputIniParserClass

### Community 93 - "CLASS solver configurations"
Cohesion: 0.40
Nodes (5): CLASS Default Settings, Fast Photometric CLASS Settings, Fast Photometric HMcode CLASS Settings, HMcode Nonlinear Model, Fast Spectroscopic CLASS Settings

### Community 94 - "Euclid spectroscopic specifications"
Cohesion: 0.40
Nodes (5): Default Nonlinear Model, Fixed Shot Noise Model, Linear-Log Spectroscopic Bias Model, Spectroscopic Galaxy Clustering Configuration, Euclid Spectroscopic DR1 Pessimistic Survey Profile

### Community 95 - "MeerKAT intensity mapping"
Cohesion: 0.40
Nodes (5): Fitting Intensity-Mapping Bias Model, Intensity Mapping Configuration, Omega HI Evolution Model, MeerKlass IM Ze Survey Profile, HI System Temperature Noise Table

### Community 96 - "SKAO intensity mapping"
Cohesion: 0.40
Nodes (5): Fitting Intensity-Mapping Bias Model, Intensity Mapping Configuration, Omega HI Evolution Model, SKAO IM Redbook Survey Profile, HI System Temperature Noise Table

### Community 97 - "SKAO spectroscopic specifications"
Cohesion: 0.40
Nodes (5): Constant Shot Noise Model, Default Nonlinear Model, Linear Spectroscopic Bias Model, Spectroscopic Galaxy Clustering Configuration, SKAO Spectroscopic Redbook Survey Profile

### Community 99 - "CMB benchmark runner"
Cohesion: 0.70
Nodes (4): main(), Path, _repo_root(), _ts()

### Community 101 - "External data installation"
Cohesion: 0.83
Nodes (3): _install_euclid_lumratio(), install_ext_data(), _install_ska_data()

### Community 102 - "Euclid q-bias specifications"
Cohesion: 0.50
Nodes (4): Constant Shot Noise Model, Default Nonlinear Model, Linear Qbias Model, Euclid Spectroscopic ISTF Qbias Survey Profile

### Community 103 - "Euclid q-bias sigma-pv"
Cohesion: 0.50
Nodes (4): Constant Shot Noise Model, Linear Qbias Model, Rescale Sigma-PV Nonlinear Model, Euclid Spectroscopic ISTF Qbias Sigma-PV Survey Profile

### Community 104 - "Euclid sigma-pv specifications"
Cohesion: 0.50
Nodes (4): Constant Shot Noise Model, Linear-Log Spectroscopic Bias Model, Rescale Sigma-PV Nonlinear Model, Euclid Spectroscopic ISTF Sigma-PV Survey Profile

### Community 105 - "SKAO photometric specifications"
Cohesion: 0.50
Nodes (4): Binned Photometric Bias Model, eNLA Intrinsic Alignment Model, Photometric Redshift Model, SKAO Photometric Redbook Survey Profile

### Community 107 - "Sampler chain metadata"
Cohesion: 0.67
Nodes (3): _format_param_label(), load_chain_metadata(), Load chain file path and sampled fiducial parameters from metadata. Args:…

### Community 109 - "Project changelog documentation"
Cohesion: 0.50
Nodes (4): Photometric Performance Optimization, Release History, Symbolic Boltzmann Solver, YAML Configuration Interface

### Community 111 - "Contribution documentation"
Cohesion: 0.50
Nodes (4): Contribution Workflow, GitHub Issues, Pull Request Checks, Sphinx Autodoc

### Community 112 - "Dark energy sampler configurations"
Cohesion: 0.50
Nodes (4): DESI w0wa Fiducial Configuration, Nested Sampling Settings, LCDM Fiducial Configuration, LCDM High H0 Fiducial Configuration

### Community 113 - "Release notes generation"
Cohesion: 0.83
Nodes (3): get_change_log_notes(), get_commit_history(), main()

### Community 115 - "Euclid one-parameter q-bias"
Cohesion: 0.67
Nodes (3): Constant Shot Noise Model, One-Parameter Linear Qbias Model, Euclid Spectroscopic ISTF Qbias One-Parameter Survey Profile

## Knowledge Gaps
- **102 isolated node(s):** `cosmicfishpie`, `release.sh script`, `run_planck_diagnostics_suite.sh script`, `CFP_ROOT`, `cosmicfishpie` (+97 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **52 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `printing` connect `Core package modules` to `Spectroscopic covariance calculations`, `Cosmology backend interface`, `Cosmological background functions`, `Spectroscopic likelihood calculations`, `Fisher matrix computation`, `Photometric angular spectra`, `FishConsumer utility functions`, `Numerical derivative calculations`, `Photometric covariance calculations`, `Spectroscopic observable setup`, `FishConsumer plotting workflow`, `Fisher parameter operations`, `CMB covariance calculations`, `Fisher comparison plots`, `Fisher plotting interface`, `Numerical helper functions`, `CMB angular spectra`, `Legacy photometric likelihood`, `Old photometric likelihood`, `CAMB photometric likelihood`?**
  _High betweenness centrality (0.264) - this node is a cross-community bridge._
- **Why does `ComputeGalSpectro` connect `Spectroscopic observable setup` to `Spectroscopic test fixtures`, `Core package modules`, `Spectroscopic array input tests`, `Spectroscopic likelihood calculations`, `Transverse Alcock-Paczynski scaling`, `Spectroscopic validation tests`, `Spectroscopic power spectrum`, `Nonlinear power spectrum`, `Spectroscopic configuration tests`, `Spectroscopic performance tests`, `Spectroscopic observable tests`, `Spectroscopic numerical stability tests`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `fisher_matrix` connect `Fisher matrix accessors` to `Fisher matrix serialization`, `Photometric benchmark tooling`, `Fisher matrix equality`, `Fisher parameter operations`, `Fisher directory comparison`, `Planck covariance comparison`, `Advanced Fisher analysis tests`, `Fisher LaTeX name setter`, `Scalar Fisher priors`, `Analysis package utilities`, `Published Planck comparison`, `Fisher LaTeX parameter names`, `Fisher fiducial parameters`, `Fisher matrix validation`, `Fisher information operations`, `Fisher comparison plots`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `ComputeGalSpectro` (e.g. with `printing` and `TestArrayHandling`) actually correct?**
  _`ComputeGalSpectro` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FisherMatrix` (e.g. with `filesystem` and `numerics`) actually correct?**
  _`FisherMatrix` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `printing` (e.g. with `FishConsumer` and `fisher_plotting`) actually correct?**
  _`printing` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cosmicfishpie`, `release.sh script`, `run_planck_diagnostics_suite.sh script` to the rest of the system?**
  _102 weakly-connected nodes found - possible documentation gaps or missing edges._
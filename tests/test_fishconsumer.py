# -*- coding: utf-8 -*-
"""
Test suite for fishconsumer module.

This module contains fast tests for the fishconsumer utility functions
and the FishConsumer class.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from cosmicfishpie.analysis import fishconsumer as fc


class TestUtilityFunctions:
    """Test utility functions in fishconsumer module."""

    def test_clamp(self):
        """Test the clamp function."""
        assert fc.clamp(100) == 100
        assert fc.clamp(300) == 255
        assert fc.clamp(-10) == 0
        assert fc.clamp(0) == 0
        assert fc.clamp(255) == 255

    def test_rgb2hex(self):
        """Test RGB to hex conversion."""
        # Test with standard RGB values
        result = fc.rgb2hex((1.0, 0.0, 0.0))  # Red
        assert result.startswith('#')
        assert len(result) == 7
        
        result = fc.rgb2hex((0.0, 1.0, 0.0))  # Green
        assert result.startswith('#')
        
        result = fc.rgb2hex((0.0, 0.0, 1.0))  # Blue
        assert result.startswith('#')

    def test_add_mathrm(self):
        """Test LaTeX formatting function."""
        # Test basic string
        result = fc.add_mathrm("Omegam")
        assert result.startswith('$')
        assert result.endswith('$')
        assert 'mathrm' in result
        
        # Test string with numbers
        result = fc.add_mathrm("sigma8")
        assert 'mathrm' in result
        
        # Test string with spaces
        result = fc.add_mathrm("test string")
        assert 'mathrm' in result

    def test_perc_to_abs(self):
        """Test percentage to absolute conversion."""
        # Test basic conversion
        result = fc.perc_to_abs(10, 100)  # 10% of 100
        assert np.isclose(result, 10)
        
        # Test with negative percentage
        result = fc.perc_to_abs(-5, 50)
        assert np.isclose(result, 2.5)
        
        # Test with zero
        result = fc.perc_to_abs(0, 100)
        assert result == 0

    def test_log_fidu_to_fidu(self):
        """Test log fiducial to fiducial conversion."""
        # Test basic conversion
        result = fc.log_fidu_to_fidu(2.0)  # 10^2 = 100
        assert np.isclose(result, 100)
        
        # Test with zero
        result = fc.log_fidu_to_fidu(0.0)  # 10^0 = 1
        assert np.isclose(result, 1)
        
        # Test with negative
        result = fc.log_fidu_to_fidu(-1.0)  # 10^-1 = 0.1
        assert np.isclose(result, 0.1)

    def test_gaussian(self):
        """Test Gaussian function."""
        # Test at the mean
        result = fc.gaussian(5, 5, 1)
        expected = 1 / (2 * 3.14159) ** 0.5
        assert np.isclose(result, expected)
        
        # Test normalization (integral should be 1)
        x = np.linspace(-10, 10, 1000)
        y = fc.gaussian(x, 0, 1)
        integral = np.trapz(y, x)
        assert np.isclose(integral, 1, rtol=1e-2)

    def test_arrays_gaussian(self):
        """Test arrays_gaussian function."""
        log_fidu = 2.0
        perc_sigma = 10.0
        xa, g1d = fc.arrays_gaussian(log_fidu, perc_sigma, nsigmas=2)
        
        # Check array properties
        assert len(xa) == len(g1d)
        assert len(xa) == 1000  # Default size
        assert np.all(g1d >= 0)  # Gaussian is always positive
        assert np.max(g1d) > 0  # Should have some positive values


class TestColorFunctions:
    """Test color-related functions."""

    def test_allnicecolors_structure(self):
        """Test that color constants are properly structured."""
        assert isinstance(fc.allnicecolors, list)
        assert len(fc.allnicecolors) > 0
        
        # Check that each element is a tuple with name and hex code
        for color in fc.allnicecolors:
            assert isinstance(color, tuple)
            assert len(color) == 2
            assert isinstance(color[0], str)  # Name
            assert isinstance(color[1], str)  # Hex code
            assert color[1].startswith('#')  # Valid hex format

    def test_allnicecolors_dict(self):
        """Test color dictionary."""
        assert isinstance(fc.allnicecolors_dict, dict)
        assert len(fc.allnicecolors_dict) > 0
        
        # Check that all values are valid hex codes
        for name, hex_code in fc.allnicecolors_dict.items():
            assert isinstance(name, str)
            assert isinstance(hex_code, str)
            assert hex_code.startswith('#')

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_display_colors(self):
        """Test display_colors function."""
        test_colors = [("Red", "#FF0000"), ("Green", "#00FF00"), ("Blue", "#0000FF")]
        
        # Test that function runs without error
        with patch('matplotlib.pyplot.show'):
            fc.display_colors(test_colors, figsize=(4, 4))


class TestDataProcessingFunctions:
    """Test data processing functions."""

    def test_n_sigmas(self):
        """Test n_sigmas function."""
        log_fidu = 1.0
        sigma_log = 0.1
        
        result = fc.n_sigmas(log_fidu, sigma_log, nsigmas=1)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] < result[1]  # Lower bound < upper bound

    def test_sigma_fidu(self):
        """Test sigma_fidu function."""
        log_fidu = 1.0
        sigma_perc_log = 10.0
        
        # Test positive sign
        result_pos = fc.sigma_fidu(log_fidu, sigma_perc_log, +1)
        result_neg = fc.sigma_fidu(log_fidu, sigma_perc_log, -1)
        
        assert result_pos > 0
        assert result_neg > 0  # Both results are positive due to the function implementation
        assert result_pos != result_neg  # But they should be different values


class TestFishConsumerClass:
    """Test the FishConsumer wrapper class."""

    def test_init(self):
        """Test FishConsumer initialization."""
        fc_instance = fc.FishConsumer()
        assert hasattr(fc_instance, 'named_colors')
        assert hasattr(fc_instance, 'barplot_colors')
        assert len(fc_instance.named_colors) > 0
        assert len(fc_instance.barplot_colors) > 0

    def test_init_with_custom_colors(self):
        """Test FishConsumer initialization with custom colors."""
        custom_colors = [("Custom", "#123456")]
        custom_barplot = [("Bar", "#654321")]
        
        fc_instance = fc.FishConsumer(
            named_colors=custom_colors,
            barplot_colors=custom_barplot
        )
        assert fc_instance.named_colors == custom_colors
        assert fc_instance.barplot_colors == custom_barplot

    def test_method_delegation(self):
        """Test that methods are properly delegated to module functions."""
        fc_instance = fc.FishConsumer()
        
        # Test clamp method
        assert fc_instance.clamp(300) == 255
        
        # Test gaussian method
        result = fc_instance.gaussian(0, 0, 1)
        expected = 1 / (2 * 3.14159) ** 0.5
        assert np.isclose(result, expected)
        
        # Test add_mathrm method
        result = fc_instance.add_mathrm("test")
        assert result.startswith('$')
        assert result.endswith('$')

    def test_perc_to_abs_method(self):
        """Test percentage to absolute conversion method."""
        fc_instance = fc.FishConsumer()
        result = fc_instance.perc_to_abs(20, 50)
        assert np.isclose(result, 10)


class TestMockFisherMatrix:
    """Test functions that work with fisher matrices using mocks."""

    def test_replace_latex_name(self):
        """Test replace_latex_name function with mock fisher matrix."""
        # Create a mock fisher matrix
        mock_fisher = Mock()
        mock_fisher.get_param_names_latex.return_value = ['\\Omega_m', 'h', '\\sigma_8']
        mock_fisher.set_param_names_latex = Mock()
        
        # Test replacement
        fc.replace_latex_name(mock_fisher, '\\Omega_m', '\\Omega_{m,0}')
        
        # Verify the method was called with updated list
        mock_fisher.set_param_names_latex.assert_called_once()
        called_args = mock_fisher.set_param_names_latex.call_args[0][0]
        assert called_args[0] == '\\Omega_{m,0}'
        assert called_args[1] == 'h'
        assert called_args[2] == '\\sigma_8'

    def test_replace_latex_style(self):
        """Test replace_latex_style function with mock fisher matrix."""
        # Create a mock fisher matrix
        mock_fisher = Mock()
        mock_fisher.get_param_names_latex.return_value = ['\\Omega_m', 'h']
        mock_fisher.set_param_names_latex = Mock()
        
        replace_dict = {'\\Omega_m': '\\Omega_{m,0}'}
        
        # Test replacement
        result = fc.replace_latex_style(mock_fisher, replace_dict)
        
        # Verify the method was called
        mock_fisher.set_param_names_latex.assert_called_once()
        assert result == mock_fisher


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestPandasIntegration:
    """Test functions that work with pandas DataFrames."""

    def test_load_Nautilus_chains_from_txt(self):
        """Test loading Nautilus chains with mock data."""
        import pandas as pd
        
        # Create temporary test data
        test_data = np.array([
            [0.3, 0.7, 1.0, -10.5],  # param1, param2, weight, posterior
            [0.31, 0.69, 0.8, -10.2],
            [0.29, 0.71, 1.2, -10.8]
        ])
        
        with patch('numpy.loadtxt', return_value=test_data):
            param_cols = ['param1', 'param2']
            result = fc.load_Nautilus_chains_from_txt('dummy.txt', param_cols)
            
            assert isinstance(result, pd.DataFrame)
            assert 'param1' in result.columns
            assert 'param2' in result.columns
            assert 'weight' in result.columns
            assert 'posterior' in result.columns
            assert len(result) == 3

    def test_load_Nautilus_chains_with_log_weights(self):
        """Test loading Nautilus chains with log weights."""
        import pandas as pd
        
        test_data = np.array([
            [0.3, 0.7, 0.0, -10.5],  # log weight = 0 -> weight = 1
            [0.31, 0.69, -0.223, -10.2],  # log weight = -0.223 -> weight ≈ 0.8
        ])
        
        with patch('numpy.loadtxt', return_value=test_data):
            param_cols = ['param1', 'param2']
            result = fc.load_Nautilus_chains_from_txt('dummy.txt', param_cols, log_weights=True)
            
            assert isinstance(result, pd.DataFrame)
            # Check that weights were exponentiated
            assert np.isclose(result['weight'].iloc[0], 1.0)
            assert result['weight'].iloc[1] < 1.0


class TestConstants:
    """Test module constants and data structures."""

    def test_barplot_filter_names(self):
        """Test barplot filter names."""
        assert isinstance(fc.barplot_filter_names, list)
        assert len(fc.barplot_filter_names) > 0
        for name in fc.barplot_filter_names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_param_latex_names(self):
        """Test parameter LaTeX names."""
        assert isinstance(fc.param_latex_names, list)
        assert isinstance(fc.parnames_style, list)
        assert len(fc.param_latex_names) == len(fc.parnames_style)

    def test_dictreplace_tex(self):
        """Test LaTeX replacement dictionary."""
        assert isinstance(fc.dictreplace_tex, dict)
        assert len(fc.dictreplace_tex) > 0
        for key, value in fc.dictreplace_tex.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestDefaultInstance:
    """Test the default FishConsumer instance."""

    def test_default_fish_consumer_exists(self):
        """Test that DEFAULT_FISH_CONSUMER exists and is properly initialized."""
        assert hasattr(fc, 'DEFAULT_FISH_CONSUMER')
        assert isinstance(fc.DEFAULT_FISH_CONSUMER, fc.FishConsumer)
        assert len(fc.DEFAULT_FISH_CONSUMER.named_colors) > 0
        assert len(fc.DEFAULT_FISH_CONSUMER.barplot_colors) > 0

    def test_default_instance_methods(self):
        """Test that default instance methods work."""
        # Test a simple method
        result = fc.DEFAULT_FISH_CONSUMER.clamp(300)
        assert result == 255
        
        # Test color access
        assert len(fc.DEFAULT_FISH_CONSUMER.named_colors) > 0
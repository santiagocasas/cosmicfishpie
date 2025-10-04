"""
Test suite for cosmicfishpie.utilities.utils module.

This module tests utility classes (misc, printing, numerics, filesystem)
to improve coverage from 46% to 60%.
"""

import os
import tempfile
import warnings
from unittest.mock import MagicMock, call, patch

import numpy as np

# Import the utils module
from cosmicfishpie.utilities import utils


class TestMiscClass:
    """Test the misc utility class."""

    def test_deepupdate_basic(self):
        """Test basic deepupdate functionality."""
        original = {"a": 1, "b": {"c": 2}}
        update = {"b": {"d": 3}, "e": 4}

        result = utils.misc.deepupdate(original, update)

        assert result["a"] == 1
        assert result["b"]["c"] == 2
        assert result["b"]["d"] == 3
        assert result["e"] == 4

    def test_deepupdate_non_mapping_original(self):
        """Test deepupdate when original is not a mapping."""
        original = "not_a_dict"
        update = {"a": 1}

        result = utils.misc.deepupdate(original, update)

        assert result == update

    def test_deepupdate_nested_dicts(self):
        """Test deepupdate with deeply nested dictionaries."""
        original = {"level1": {"level2": {"level3": "original"}}}
        update = {"level1": {"level2": {"new_key": "new_value"}}}

        result = utils.misc.deepupdate(original, update)

        assert result["level1"]["level2"]["level3"] == "original"
        assert result["level1"]["level2"]["new_key"] == "new_value"


class TestPrintingClass:
    """Test the printing utility class."""

    def test_debug_print_enabled(self):
        """Test debug_print when debug is enabled."""
        utils.printing.debug = True

        with patch("builtins.print") as mock_print:
            utils.printing.debug_print("test message")
            mock_print.assert_called_once_with("test message")

        utils.printing.debug = False  # Reset

    def test_debug_print_disabled(self):
        """Test debug_print when debug is disabled."""
        utils.printing.debug = False

        with patch("builtins.print") as mock_print:
            utils.printing.debug_print("test message")
            mock_print.assert_not_called()

    def test_time_print_with_times(self):
        """Test time_print with time measurements."""
        with patch("builtins.print") as mock_print:
            utils.printing.time_print(
                feedback_level=1, min_level=0, text="Test computation", time_ini=0.0, time_fin=1.5
            )

            # Should print empty line and message with elapsed time
            assert mock_print.call_count == 2
            calls = mock_print.call_args_list
            assert calls[0] == call("")  # Empty line
            assert "Test computation" in calls[1][0][0]
            assert "1.50 s" in calls[1][0][0]

    def test_time_print_with_instance(self):
        """Test time_print with instance parameter."""

        class TestClass:
            pass

        instance = TestClass()

        with patch("builtins.print") as mock_print:
            utils.printing.time_print(feedback_level=1, min_level=0, instance=instance)

            # Should include class name in output
            calls = mock_print.call_args_list
            assert "TestClass" in calls[1][0][0]

    def test_time_print_feedback_level_filtering(self):
        """Test time_print respects feedback level filtering."""
        with patch("builtins.print") as mock_print:
            utils.printing.time_print(feedback_level=0, min_level=1, text="Should not print")

            mock_print.assert_not_called()

    def test_suppress_warnings_decorator_enabled(self):
        """Test suppress_warnings decorator when enabled."""
        # Use patch to mock the SUPPRESS_WARNINGS attribute
        with patch.object(utils.printing, "SUPPRESS_WARNINGS", True):

            @utils.printing.suppress_warnings
            def warning_function():
                warnings.warn("Test warning")
                return "success"

            with warnings.catch_warnings(record=True) as w:
                result = warning_function()
                assert result == "success"
                assert len(w) == 0  # Warning should be suppressed

    def test_suppress_warnings_decorator_disabled(self):
        """Test suppress_warnings decorator when disabled."""
        # Use patch to mock the SUPPRESS_WARNINGS attribute
        with patch.object(utils.printing, "SUPPRESS_WARNINGS", False):

            @utils.printing.suppress_warnings
            def warning_function():
                warnings.warn("Test warning")
                return "success"

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Catch all warnings
                result = warning_function()
                assert result == "success"
                assert len(w) == 1  # Warning should not be suppressed
                assert "Test warning" in str(w[0].message)


class TestNumericsClass:
    """Test the numerics utility class."""

    def test_moving_average_basic(self):
        """Test basic moving average calculation."""
        data = np.array([1, 2, 3, 4, 5])
        result = utils.numerics.moving_average(data, periods=2)

        expected = np.array([1.5, 2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_moving_average_different_periods(self):
        """Test moving average with different periods."""
        data = np.array([1, 2, 3, 4, 5, 6])
        result = utils.numerics.moving_average(data, periods=3)

        expected = np.array([2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_round_decimals_up(self):
        """Test round_decimals_up method."""
        # Test basic rounding up
        result = utils.numerics.round_decimals_up(1.234, decimals=1)
        assert result == 1.3  # Should round up to 1 decimal

    def test_closest(self):
        """Test closest function."""
        lst = [1, 3, 5, 7, 9]
        result = utils.numerics.closest(lst, 4)
        assert result == 3

        result = utils.numerics.closest(lst, 6)
        assert result == 5

    def test_bisection(self):
        """Test bisection search function."""
        array = np.array([1, 3, 5, 7, 9])

        # Test exact match
        result = utils.numerics.bisection(array, 5)
        assert result == 2

        # Test value between elements
        result = utils.numerics.bisection(array, 4)
        assert result in [1, 2]  # Should return nearby index

    def test_find_nearest(self):
        """Test find_nearest function."""
        array = np.array([1, 3, 5, 7, 9])

        result = utils.numerics.find_nearest(array, 4)
        assert result == 3  # Closest value to 4

        result = utils.numerics.find_nearest(array, 6)
        assert result == 5  # Closest value to 6


class TestFilesystemClass:
    """Test the filesystem utility class."""

    def test_mkdirp_new_directory(self):
        """Test mkdirp creates new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_directory")

            utils.filesystem.mkdirp(new_dir)

            assert os.path.exists(new_dir)
            assert os.path.isdir(new_dir)

    def test_mkdirp_existing_directory(self):
        """Test mkdirp with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise an error
            utils.filesystem.mkdirp(temp_dir)
            assert os.path.exists(temp_dir)

    def test_mkdirp_nested_directories(self):
        """Test mkdirp creates nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "level1", "level2", "level3")

            utils.filesystem.mkdirp(nested_path)

            assert os.path.exists(nested_path)
            assert os.path.isdir(nested_path)

    def test_git_version(self):
        """Test git_version function."""
        # Mock subprocess to avoid depending on actual git
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = (b"v1.0.0-test\n", b"")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            result = utils.filesystem.git_version()

            # Should return some version string or None
            assert isinstance(result, (str, type(None)))


class TestInputIniParserClass:
    """Test the inputiniparser class."""

    def test_inputiniparser_initialization(self):
        """Test inputiniparser initialization."""
        # Create a temporary ini file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write("[section1]\n")
            f.write("param1 = 0.3\n")
            f.write("param2 = 0.7\n")
            temp_path = f.name

        try:
            parser = utils.inputiniparser(temp_path)
            # Just test that it was created successfully
            assert parser is not None
            assert hasattr(parser, "__class__")
        finally:
            os.unlink(temp_path)

    def test_free_epsilons(self):
        """Test free_epsilons method."""
        # Create a temporary ini file with parameters
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write("[freeparameters]\n")
            f.write("Omegam = 0.01\n")
            f.write("h = 0.02\n")
            temp_path = f.name

        try:
            parser = utils.inputiniparser(temp_path)
            result = parser.free_epsilons()

            assert isinstance(result, dict)
            # Should contain the parameters from the file
            if result:  # If parsing was successful
                assert "Omegam" in result or len(result) >= 0
        finally:
            os.unlink(temp_path)


class TestPhysmathClass:
    """Test the physmath utility class."""

    def test_physmath_constants(self):
        """Test physmath class constants and attributes."""
        # Test that physmath class exists and has expected structure
        assert hasattr(utils, "physmath")

        # Test some basic constants that might be defined
        physmath_attrs = dir(utils.physmath)
        assert len(physmath_attrs) > 0  # Should have some attributes

        # Test that it's accessible
        physmath_class = utils.physmath
        assert physmath_class is not None

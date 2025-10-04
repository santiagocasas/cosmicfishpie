"""
Test suite for cosmicfishpie.analysis.colors module.

This module tests color utility functions to improve coverage.
"""

from cosmicfishpie.analysis.colors import bash_colors, nice_colors


class TestNiceColors:
    """Test the nice_colors function."""

    def test_nice_colors_integer_inputs(self):
        """Test nice_colors with integer inputs."""
        # Test all defined colors (0-6)
        for i in range(7):
            color = nice_colors(i)
            assert isinstance(color, tuple)
            assert len(color) == 3  # RGB tuple

            # All values should be between 0 and 1
            for component in color:
                assert 0 <= component <= 1
                assert isinstance(component, float)

    def test_nice_colors_float_inputs(self):
        """Test nice_colors with float inputs."""
        # Test with float inputs
        color = nice_colors(2.7)
        assert isinstance(color, tuple)
        assert len(color) == 3

        color = nice_colors(0.1)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_nice_colors_modulo_behavior(self):
        """Test that nice_colors uses modulo 7."""
        # Test that values beyond 6 wrap around
        color_0 = nice_colors(0)
        color_7 = nice_colors(7)
        color_14 = nice_colors(14)

        assert color_0 == color_7 == color_14

        color_3 = nice_colors(3)
        color_10 = nice_colors(10)

        assert color_3 == color_10

    def test_nice_colors_negative_inputs(self):
        """Test nice_colors with negative inputs."""
        # Negative numbers should still work with modulo
        color = nice_colors(-1)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_nice_colors_specific_values(self):
        """Test specific known color values."""
        # Test that the first color matches the expected value
        color_0 = nice_colors(0)
        expected_0 = (203.0 / 255.0, 15.0 / 255.0, 40.0 / 255.0)

        # Check each component with tolerance for floating point precision
        for actual, expected in zip(color_0, expected_0):
            assert abs(actual - expected) < 1e-6

    def test_nice_colors_all_different(self):
        """Test that all defined colors are different."""
        colors = [nice_colors(i) for i in range(7)]

        # Check that all colors are unique
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                assert colors[i] != colors[j]


class TestBashColors:
    """Test the bash_colors class."""

    def setUp(self):
        """Set up test fixtures."""
        self.bash_color = bash_colors()

    def test_bash_colors_initialization(self):
        """Test bash_colors class initialization."""
        bc = bash_colors()
        assert isinstance(bc, bash_colors)

        # Check that all ANSI codes are strings
        assert isinstance(bc.HEADER, str)
        assert isinstance(bc.OKBLUE, str)
        assert isinstance(bc.OKGREEN, str)
        assert isinstance(bc.WARNING, str)
        assert isinstance(bc.FAIL, str)
        assert isinstance(bc.BOLD, str)
        assert isinstance(bc.UNDERLINE, str)
        assert isinstance(bc.ENDC, str)

    def test_bash_colors_constants(self):
        """Test that ANSI color constants are defined correctly."""
        bc = bash_colors()

        # Check that constants start with ANSI escape sequence
        assert bc.HEADER.startswith("\033[")
        assert bc.OKBLUE.startswith("\033[")
        assert bc.OKGREEN.startswith("\033[")
        assert bc.WARNING.startswith("\033[")
        assert bc.FAIL.startswith("\033[")
        assert bc.BOLD.startswith("\033[")
        assert bc.UNDERLINE.startswith("\033[")
        assert bc.ENDC.startswith("\033[")

    def test_header_method(self):
        """Test the header method."""
        bc = bash_colors()
        test_string = "Test Header"
        result = bc.header(test_string)

        assert isinstance(result, str)
        assert test_string in result
        assert bc.HEADER in result
        assert bc.ENDC in result
        assert result.startswith(bc.HEADER)
        assert result.endswith(bc.ENDC)

    def test_blue_method(self):
        """Test the blue method."""
        bc = bash_colors()
        test_string = "Test Blue"
        result = bc.blue(test_string)

        assert isinstance(result, str)
        assert test_string in result
        assert bc.OKBLUE in result
        assert bc.ENDC in result

    def test_green_method(self):
        """Test the green method."""
        bc = bash_colors()
        test_string = "Test Green"
        result = bc.green(test_string)

        assert isinstance(result, str)
        assert test_string in result
        assert bc.OKGREEN in result
        assert bc.ENDC in result

    def test_warning_method(self):
        """Test the warning method."""
        bc = bash_colors()
        test_string = "Test Warning"
        result = bc.warning(test_string)

        assert isinstance(result, str)
        assert test_string in result
        assert bc.WARNING in result
        assert bc.ENDC in result

    def test_fail_method(self):
        """Test the fail method."""
        bc = bash_colors()
        test_string = "Test Fail"
        result = bc.fail(test_string)

        assert isinstance(result, str)
        assert test_string in result
        assert bc.FAIL in result
        assert bc.ENDC in result

    def test_bold_method(self):
        """Test the bold method."""
        bc = bash_colors()
        test_string = "Test Bold"
        result = bc.bold(test_string)

        assert isinstance(result, str)
        assert test_string in result
        assert bc.BOLD in result
        assert bc.ENDC in result

    def test_underline_method(self):
        """Test the underline method."""
        bc = bash_colors()
        test_string = "Test Underline"
        result = bc.underline(test_string)

        assert isinstance(result, str)
        assert test_string in result
        assert bc.UNDERLINE in result
        assert bc.ENDC in result

    def test_all_methods_with_numbers(self):
        """Test all color methods with numeric inputs."""
        bc = bash_colors()
        test_number = 42

        methods = [bc.header, bc.blue, bc.green, bc.warning, bc.fail, bc.bold, bc.underline]

        for method in methods:
            result = method(test_number)
            assert isinstance(result, str)
            assert str(test_number) in result

    def test_all_methods_with_empty_string(self):
        """Test all color methods with empty string."""
        bc = bash_colors()
        test_string = ""

        methods = [bc.header, bc.blue, bc.green, bc.warning, bc.fail, bc.bold, bc.underline]

        for method in methods:
            result = method(test_string)
            assert isinstance(result, str)
            # Should still contain ANSI codes even with empty string
            assert len(result) > 0

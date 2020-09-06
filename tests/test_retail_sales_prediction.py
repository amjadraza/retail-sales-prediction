#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `retail_sales_prediction` package."""
import sys
sys.path.append('.')

import unittest
from click.testing import CliRunner

from retail_sales_prediction import cli


class TestRetail_sales_prediction(unittest.TestCase):
    """Tests for `retail_sales_prediction` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'retail_sales_prediction.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

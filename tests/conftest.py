# tests/conftest.py

"""
Global fixtures and configurations for the test suite.

This file provides shared utilities, fixtures, and hooks for pytest.
It automatically adds a --verbose-filters command-line option to control
the verbosity of filter state printing during tests.
"""

import pytest
from typing import Callable
from libFilter.core.base import FilterBase

def pytest_addoption(parser):
    """Adds a custom command-line option to pytest."""
    parser.addoption(
        "--verbose-filters", action="store_true", default=False,
        help="Print detailed filter state during test execution."
    )

@pytest.fixture
def verbose_printer(request) -> Callable[[FilterBase, str], None]:
    """
    A fixture that provides a function to print filter state if the
    --verbose-filters flag is set.
    """
    is_verbose = request.config.getoption("--verbose-filters")

    def _printer(filter_instance: FilterBase, stage: str):
        """Prints filter state if verbosity is enabled."""
        if is_verbose:
            print(f"\n--- [State after: {stage}] ---\n{filter_instance.to_string(verbose=False)}")
    
    return _printer
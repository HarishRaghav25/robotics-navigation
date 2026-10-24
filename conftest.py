"""
Root pytest configuration and fixtures for robotics navigation test suite.
Ensures repository root is included in sys.path so tests can run without PYTHONPATH=.
Also ensures the current working directory is anchored to the project root during tests.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True, scope="session")
def set_project_root_cwd():
    """Anchor working directory to repository root across all test runs."""
    orig_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield
    try:
        os.chdir(orig_cwd)
    except Exception:
        pass


@pytest.fixture
def project_root() -> Path:
    """Fixture returning the resolved project root directory."""
    return PROJECT_ROOT

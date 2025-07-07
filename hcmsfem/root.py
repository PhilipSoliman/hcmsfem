import sys
from os.path import abspath, dirname
from pathlib import Path


def get_source_root() -> Path:
    file_abs_path = abspath(dirname(__file__))
    return Path(file_abs_path)


def get_package_root() -> Path:
    return get_source_root().parent


def get_venv_root() -> Path:
    """
    Find the parent directory of the virtual environment folder from which Python is currently running.
    Gets the third parent of the Python executable location.
    """
    python_executable = Path(sys.executable)
    return python_executable.parent.parent.parent

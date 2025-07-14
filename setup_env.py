import argparse
import os
import re
import subprocess
import sys
import venv

# Get the absolute path of the script directory 
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Get the absolute path of the local package directory (same as script directory)
PACKAGE_DIR = SCRIPT_DIR

# Global variables that will be set based on command line arguments
VENV_DIR = None
PYTHON_EXEC = None

# CUDA versions supported by PyTorch
CUDA_VERSIONS = ["cu118", "cu126", "cu128"]


def setup_paths(use_parent=False, venv_name=".venv"):
    """Setup global path variables based on whether to use parent directory and venv name."""
    global VENV_DIR, PYTHON_EXEC

    if use_parent:
        # Install venv in parent of workspace root
        VENV_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), venv_name)
    else:
        # Install venv in workspace root (default behavior)
        VENV_DIR = os.path.join(SCRIPT_DIR, venv_name)

    PYTHON_EXEC = (
        os.path.join(VENV_DIR, "Scripts", "python.exe")
        if os.name == "nt"
        else os.path.join(VENV_DIR, "bin", "python")
    )


def create_virtual_environment():
    if not os.path.exists(VENV_DIR):
        venv.create(VENV_DIR, with_pip=True)
        print(f"✅ Created virtual environment '{VENV_DIR}'\n")
    else:
        print(f"Virtual environment '{VENV_DIR}' already exists")


def install_package():

    # upgrade pip
    subprocess.check_call([PYTHON_EXEC, "-m", "pip", "install", "--upgrade", "pip"])
    print("✅ Upgraded pip to latest version\n")

    # install local python package
    subprocess.check_call(
        [
            PYTHON_EXEC,
            "-m",
            "pip",
            "install",
            "-e",
            PACKAGE_DIR,
        ]
    )
    print(f"✅ Installed local package in {PACKAGE_DIR}\n")

    # install PyTorch with or without CUDA support
    cuda_version = _get_cuda_version()
    cuda_used = False
    print(f"Detected CUDA version: {cuda_version}")
    if cuda_version in CUDA_VERSIONS:
        cuda_used = True
        torch_url = f"https://download.pytorch.org/whl/{cuda_version}"
    else:
        torch_url = "https://download.pytorch.org/whl/cpu"

    subprocess.check_call(
        [
            PYTHON_EXEC,
            "-m",
            "pip",
            "install",
            "torch",
            "--index-url",
            torch_url,
        ]
    )
    print(
        f"✅ Installed PyTorch {'with CUDA support' if cuda_used else 'without CUDA support'}\n"
    )


def generate_meshes_for_experiments():
    mesh_script = os.path.join(PACKAGE_DIR, "generate_meshes.py")
    user_input = input(f"Generate meshes for experiments? Yes/No [y/n]: ")

    if user_input.lower() != "y":
        print("Skipping mesh generation.")
        return

    result = subprocess.run(
        [PYTHON_EXEC, mesh_script],
        cwd=PACKAGE_DIR,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        print("Failed to generate meshes. Exiting.")
        sys.exit(1)

    print("✅ Meshes generated successfully\n")


def activate_environment():
    subprocess.run(
        [PYTHON_EXEC, os.path.join(PACKAGE_DIR, "activate_env.py")],
        shell=True,
        check=True,
    )


def _get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in output.split("\n"):
            if "release" in line:
                match = re.search(r"release (\d+)\.(\d+)", line)
                if match:
                    major, minor = match.groups()
                    version = f"{major}{minor}"
                    return f"cu{version}"
                else:
                    raise ValueError("Could not parse CUDA version from nvcc output.")
    except FileNotFoundError:
        input(
            "CUDA not found. If you want to speed up some calculations in this project, please install CUDA. Press Enter to continue without CUDA."
        )
        return ""


def setup_vscode_settings():
    """Setup VSCode settings by calling the setup_vscode.py script."""
    setup_vscode_script = os.path.join(SCRIPT_DIR, "setup_vscode.py")

    if not os.path.exists(setup_vscode_script):
        print(f"Warning: VSCode setup script not found at {setup_vscode_script}")
        return

    # Ask user if they want to setup VSCode settings
    user_input = input("Setup VSCode settings for this project? [y/n]: ")
    if user_input.lower() != "y":
        print("Skipping VSCode setup.")
        return

    # Determine target directory (same as VENV_DIR parent)
    target_dir = os.path.dirname(VENV_DIR)

    try:
        # Call setup_vscode.py as subprocess
        subprocess.check_call(
            [
                sys.executable,
                setup_vscode_script,
                "--dir",
                target_dir,
                "--force",  # Don't prompt since we already asked the user
            ]
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to setup VSCode settings: {e}")
    except Exception as e:
        print(f"Error running VSCode setup: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Setup Python environment for hcmsfem project"
    )
    parser.add_argument(
        "--parent",
        action="store_true",
        help=(
            "Install virtual environment in parent directory of workspace root."
            " NOTE: the 'logs/', 'data/' and 'figures/' directories will be created in the same directory as the virtual environment."
        ),
    )
    parser.add_argument(
        "--venv-name",
        default=".venv",
        help="Name of the virtual environment directory (default: .venv)",
    )
    parser.add_argument(
        "--skip-vscode",
        action="store_true",
        help="Skip VSCode settings setup",
    )

    args = parser.parse_args()

    # Setup paths based on CLI arguments
    setup_paths(use_parent=args.parent, venv_name=args.venv_name)

    # Print where the virtual environment will be created
    abs_venv_path = os.path.abspath(VENV_DIR)
    print(f"Virtual environment will be created at: {abs_venv_path}")

    create_virtual_environment()
    install_package()
    generate_meshes_for_experiments()

    # Setup VSCode settings (unless skipped)
    if not args.skip_vscode:
        setup_vscode_settings()

    activate_environment()
    sys.exit(0)


if __name__ == "__main__":
    main()

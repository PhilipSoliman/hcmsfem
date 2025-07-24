# High Contrast Multiscale Finite Element Method (HCMSFEM)
This section outlines the setup process for the project, including the installation of [required software](#requirements) and configuration of the development environment.

## Requirements
Check that you have the following 
- `python` (3.10 or higher)
- Full [TeXLive](https://www.tug.org/texlive/windows.html) installation. On Linux this can be done using the following command:
```bash
apt-get install texlive-full
```
possibly using `sudo`. If you are on Windows, the above should be made available in your PATH. You can check this by running the following command in a terminal:
```bash
tex --version
```
Lastly for the LaTeX compilation it is necessary that the path to this repository does not contain any spaces. This is a [known issue](https://github.com/James-Yu/LaTeX-Workshop/issues/2910) with the `latexmk` tool, which is used for compiling the LaTeX files.

---

## Setup Script
Firstly, the [HCMSFEM](https://github.com/PhilipSoliman/hcmsfem) repository is set up as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) of this repository. If you have not done so already, run the following command after cloning:
```bash
git submodule update --init
```
This clones HCMSFEM repository at the specific commit on the [philip-soliman-am-master-thesis](https://github.com/PhilipSoliman/hcmsfem/tree/philip-soliman-am-master-thesis) branch from which this main repository benefits. For more information on how to use git submodules, refer to the [Git Submodules documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

Second, use the provided [setup_env.py](hcmsfem/setup_env.py) script. On Linux, you might need to give it rights by running the following command (in a bash shell):
```bash
chmod +x hcmsfem/setup_env.py
```
The script automatically sets up a virtual environment, installs hcmsfem and all requirements it relies on; listed in its [pyproject.toml](hcmsfem/pyproject.toml) file. Simply use your python installation to run the script:
```bash
<python-executable> hcmsfem/setup_env.py
```
The virtual environment, logs, data and figures folders are created in this repository's root. For more setup options, run the script with the `--help` flag to see all available options:
```bash
<python-executable> hcmsfem/setup_env.py --help
```
On Windows, the `setup_env.py` script will also activate the environment for you, while on Linux you will need to explicitly do so by running the following command (in a bash shell):
```bash
source .venv/bin/activate
```

### VSCode Extensions
After the setup script is run and in case some you do not already have them installed, you should be prompted to install the following VSCode extensions:
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
- [Action Buttons](https://marketplace.visualstudio.com/items?itemName=seunlanlege.action-buttons)

For the Action Buttons to appear, you will need to press the white "Reload" button in the bottom left corner of your VSCode Workspace, after installing the extension.

## Running Code
Any python script created in this repository can be run using the python interpreter from the virtual environment created by the setup script. To run a script, simply use the following command in the terminal (after the environment is activated):
```bash
python <script_name>.py
```
Or, select the virtual environment as the current workspace's python interpreter in VSCode and run the script using the "Run Python File in Terminal" command or play button. This will ensure that the script is run with the correct python interpreter and all dependencies are available.

Any script that imports the `hcmsfem` package can be run with the `--help` flag to see all available options. For example, to run the experiment that generates all quadrilateral meshes for experiments with info logging and progress bar, use the following command:
```bash
python hcmsfem/generate_meshes.py --loglvl info --show-progress
```
Or, to run the [experiment](code/model_spectra/approximate_spectra.py) that calculates spectra for the GDSW, RGDSW and AMS preconditioners for a diffusion problem with HDBCs on a unit square for various high-contrast coefficient functions with debug logging and progress bar, use the following command:
```bash
python code/model_spectra/approximate_spectra.py --loglvl debug --show-progress
```

## Showing and Generating Figures
Any file in the [code](code) folder that ends with `_fig.py` is a script that generates a figure. These scripts can be run using the python interpreter from the virtual environment created by the setup script. To run a figure generating script and show its output, simply use the following command in the terminal (after the environment is activated):
```bash
python path_to_script/*_fig.py --show-output
```
Or, to generate an output PDF file without showing it, use the following command:
```bash
python path_to_script/*_fig.py --generate-output
```
The above actions can also be done by clicking on the action buttons that are configured automatically by the [setup script](#vscode-extensions).

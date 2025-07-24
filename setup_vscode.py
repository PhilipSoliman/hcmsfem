#!/usr/bin/env python3
"""
Standalone script to setup VSCode settings and extensions for the thesis project.

This script creates a .vscode folder with recommended settings and extensions
for working with this LaTeX/Python thesis project. It loads the configuration
from vscode_settings.json in the same directory.
"""

import json
import os
import sys


def load_vscode_config():
    """Load VSCode configuration from vscode_settings.json."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(script_dir, "vscode_settings.json")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"VSCode configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_vscode_settings(target_dir=None, force=False):
    """Setup VSCode settings and recommend extensions."""
    # Load configuration
    try:
        config = load_vscode_config()
        vscode_settings = config["settings"]
        vscode_extensions = config["extensions"]
        vscode_launch = config["launch"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading VSCode configuration: {e}")
        return False

    # Determine target directory
    if target_dir:
        # Use provided directory
        vscode_dir = os.path.join(target_dir, ".vscode")
    else:
        # Default: project root (one level up from hcmsfem)
        script_dir = os.path.abspath(os.path.dirname(__file__))
        vscode_dir = os.path.join(script_dir, ".vscode")

    settings_file = os.path.join(vscode_dir, "settings.json")
    extensions_file = os.path.join(vscode_dir, "extensions.json")
    launch_file = os.path.join(vscode_dir, "launch.json")

    print("Setting up VSCode configuration for thesis project...")
    print(f"Target directory: {vscode_dir}")

    # Create .vscode directory if it doesn't exist
    os.makedirs(vscode_dir, exist_ok=True)

    # Handle settings.json
    if os.path.exists(settings_file) and not force:
        user_input = input("VSCode settings.json already exists. Overwrite? [y/n]: ")
        if user_input.lower() != "y":
            print("Keeping existing settings.json")
        else:
            _write_settings_file(settings_file, vscode_settings)
    else:
        _write_settings_file(settings_file, vscode_settings)

    # Handle extensions.json
    if os.path.exists(extensions_file) and not force:
        user_input = input("VSCode extensions.json already exists. Overwrite? [y/n]: ")
        if user_input.lower() != "y":
            print("Keeping existing extensions.json")
        else:
            _write_extensions_file(extensions_file, vscode_extensions)
    else:
        _write_extensions_file(extensions_file, vscode_extensions)

    # Handle launch.json
    if os.path.exists(launch_file) and not force:
        user_input = input("VSCode launch.json already exists. Overwrite? [y/n]: ")
        if user_input.lower() != "y":
            print("Keeping existing launch.json")
        else:
            _write_launch_file(launch_file, vscode_launch)
    else:
        _write_launch_file(launch_file, vscode_launch)

    print("✅ VSCode setup complete!")
    print("Recommended next steps:")
    print("1. Open this project in VSCode")
    print("2. Install recommended extensions when prompted")
    print(
        "3. Press the white reload button (↻) in the bottom status bar to load action buttons"
    )
    return True


def _write_settings_file(settings_file, settings):
    """Write the settings.json file."""
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    print(f"✅ Created VSCode settings at: {settings_file}")


def _write_extensions_file(extensions_file, extensions):
    """Write the extensions.json file with recommended extensions."""
    extensions_config = {"recommendations": extensions}
    with open(extensions_file, "w", encoding="utf-8") as f:
        json.dump(extensions_config, f, indent=2)
    print(f"✅ Created VSCode extensions recommendations at: {extensions_file}")


def _write_launch_file(launch_file, launch_config):
    """Write the launch.json file with debug configurations."""
    with open(launch_file, "w", encoding="utf-8") as f:
        json.dump(launch_config, f, indent=2)
    print(f"✅ Created VSCode launch configurations at: {launch_file}")


def main():
    """Main function to setup VSCode configuration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup VSCode settings and extensions for the thesis project"
    )
    parser.add_argument(
        "--dir",
        help=("Directory where to create the .vscode folder." 
              " Can also be a relative path to this script."
              " For a .vscode folder in the current directory write '--dir .'" 
              " (default: same as this script)"),
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files without prompting",
    )

    args = parser.parse_args()

    success = setup_vscode_settings(target_dir=args.dir, force=args.force)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

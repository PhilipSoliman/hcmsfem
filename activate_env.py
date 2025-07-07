import os
import subprocess


def activate_environment(venv_name=".venv"):
    if os.name == "nt":
        activation_script = os.path.join(venv_name, "Scripts", "activate.bat")
        input(
            "Press Enter to activate the environment.",
        )
        subprocess.run(["cmd.exe", "/k", activation_script])
    else:
        activation_script = f"{venv_name}/bin/activate"
        print("To activate the environment, run:")
        print(f"source {activation_script}", end="", flush=True)


if __name__ == "__main__":
    activate_environment()

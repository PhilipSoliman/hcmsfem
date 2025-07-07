import argparse

# get CLI
TOGGLE_FLAGS = ["generate-output", "show-output", "show-progress"]
VALUE_FLAGS = ["loglvl"]


def get_cli_args(
    toggle_flags: list[str] = TOGGLE_FLAGS, value_flags: list[str] = VALUE_FLAGS
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # # Positional argument
    # parser.add_argument("name", help="Your name")

    # # Optional argument
    # parser.add_argument("--age", type=int, help="Your age", default=18)

    # Flag argument
    for flag in toggle_flags:
        parser.add_argument(
            f"--{flag}", action="store_true", help=f"Enable {flag} mode"
        )

    # Value flags (store value)
    for flag in value_flags:
        if flag == "loglvl":
            parser.add_argument(
                f"--{flag}",
                type=lambda s: s.upper(),
                help=f"Set logging level (e.g. --{flag} info) options: debug, info, warning, error, critical",
                default="INFO",
            )
        else:
            parser.add_argument(
                f"--{flag}",
                type=str,
                help=f"Set {flag.replace('-', ' ')} (e.g. --{flag} ...)",
                default=None,
            )

    args = parser.parse_args()
    return args

CLI_ARGS = get_cli_args()
# NOTE: Before anything, set environement variables for torch and scikit-learn to avoid issues with multiple threads
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from hcmsfem.logger import LOGGER, PROGRESS

try:
    # look for command line arguments to set the log level
    from hcmsfem.cli import CLI_ARGS

    if hasattr(CLI_ARGS, "loglvl"):
        LOGGER.setLevel(CLI_ARGS.loglvl)
    if hasattr(CLI_ARGS, "show_progress") and CLI_ARGS.show_progress:
        LOGGER.debug("Showing progress bar (CLI argument)")
        PROGRESS.show()
    else:
        LOGGER.debug("Hiding progress bar (CLI argument)")
        PROGRESS.hide()
except Exception as e:
    LOGGER.exception(f"Failed to get CLI arguments: {e}")
    pass

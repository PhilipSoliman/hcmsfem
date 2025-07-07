import inspect
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from hcmsfem.root import get_venv_root

ROOT = get_venv_root()


##########
# LOGGER #
##########
# Helper function to format file paths for logging
def format_fp(fp: Optional[Path]) -> str:
    """Format a file path for logging."""
    if fp is None:
        return ""
    # Return the path relative to ROOT if possible, else absolute path
    try:
        return "...[bold]project root[/bold]\\" + str(fp.relative_to(ROOT))
    except ValueError:
        return str(fp)


# Custom logger class to handle Path formatting and custom log levels
class CustomLogger(logging.Logger):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    # Setup log directory and file
    LOG_DIR = ROOT / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ) -> None:
        # find Paths in args and format them
        args_new = []
        for arg in args:
            if isinstance(arg, Path):
                arg = format_fp(arg)
            args_new.append(arg)
        super()._log(
            level,
            msg,
            tuple(args_new),
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel,
        )

    def setLevel(self, level: int | str) -> None:
        """Set the logging level for the logger."""
        super().setLevel(level)
        # also set the level for all handlers
        for handler in self.handlers:
            handler.setLevel(level)
    
    @contextmanager
    def setLevelContext(self, level: int | str):
        """Context manager to temporarily set the logging level."""
        original_level = self.level
        self.setLevel(level)
        try:
            yield
        finally:
            self.setLevel(original_level)

    def generate_log_file(
        self,
        level: int = logging.DEBUG,
    ) -> None:
        # Get the calling script's filename
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        script_name = "unknown"
        if module and hasattr(module, "__file__"):
            script_name = Path(module.__file__).stem

        # construct the log filename with timestamp
        log_filename = (
            self.LOG_DIR / f"{script_name}_run_{datetime.now():%Y%m%d_%H%M%S}.log"
        )

        # Create a file handler with rotation
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )

        # set log level for the file handler
        file_handler.setLevel(level)

        # # set log file format the same as the main logger first handler's formatter or a default one
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(formatter)

        # add the file handler to the logger
        self.addHandler(file_handler)


# Custom formatter to add colors to log levels
class ColorLevelFormatter(logging.Formatter):
    LEVEL_NAME_WIDTH = 8
    ERROR_COLOR = "red"
    WARNING_COLOR = "yellow"
    INFO_COLOR = "green"
    DEBUG_COLOR = "blue"
    LEN_TAB = 4
    TAB_FORMAT = f"|{'-'* (LEN_TAB - 2)}>"

    def format(self, record) -> str:
        record.levelcolor = self.get_color(record)
        emoji = self.get_emoji(record)
        record.levelname = f"{(emoji + record.levelname):<{self.LEVEL_NAME_WIDTH-1}}"
        prefix = self.get_prefix(record)
        message = record.getMessage()
        lines = message.splitlines()
        formatted_message = ""
        if len(lines) > 0:
            formatted_message = f"{prefix}{lines[0]}"
            if len(lines) > 1:
                beginning = f"\n{'':<{self.LEVEL_NAME_WIDTH+1}}{'':<{len(prefix)}}"
                for line in lines[1:]:
                    line = line.replace("\t", self.TAB_FORMAT)
                    line = f"{beginning}{line}"
                    formatted_message += line
        record.message = formatted_message
        return self.formatMessage(record)

    def get_color(self, record):
        if record.levelno == logging.ERROR:
            return self.ERROR_COLOR
        elif record.levelno == logging.WARNING:
            return self.WARNING_COLOR
        elif record.levelno == logging.INFO:
            return self.INFO_COLOR
        elif record.levelno == logging.DEBUG:
            return self.DEBUG_COLOR
        else:
            return "white"

    def get_emoji(self, record):
        """Get the emoji for the log record."""
        if record.levelno == logging.ERROR:
            return "❌ "  # exclamation mark emoji
        elif record.levelno == logging.WARNING:
            return "⚠️  "  # warning emoji
        elif record.levelno == logging.INFO:
            return "✅ "  # information emoji simple
        elif record.levelno == logging.DEBUG:
            return "🧿 "  # bug emoji
        else:
            return ""

    def get_prefix(self, record):
        """Get the prefix for the log record."""
        if record.levelno == logging.DEBUG:
            return f"➥{'':<{self.LEN_TAB}}"
        elif record.levelno == logging.WARNING:
            return " "
        else:
            return ""


# shorten warn name
logging.addLevelName(logging.WARNING, "WARN")

# instantiate the custom logger
logging.setLoggerClass(CustomLogger)
LOGGER: CustomLogger = logging.getLogger("lib")  # type: ignore[assignment]
LOGGER = CustomLogger("lib")  # type: ignore[assignment]

# set rich text handler
handler = RichHandler(
    rich_tracebacks=True,
    markup=True,
    show_level=False,
    show_time=True,
    show_path=True,
    log_time_format="[%Y-%m-%d %H:%M:%S.%f]",
)

handler.setFormatter(
    ColorLevelFormatter("[%(levelcolor)s]%(levelname)s %(message)s[/%(levelcolor)s]")
)
LOGGER.addHandler(handler)

# set default log level
if os.environ.get("MY_LOG_LEVEL"):
    LOGGER.setLevel(os.environ["MY_LOG_LEVEL"].upper())
    LOGGER.debug(
        f"Log level set to {os.environ['MY_LOG_LEVEL'].upper()} via environment variable"
    )
else:
    LOGGER.setLevel(logging.INFO)


################
# PROGRESS BAR #
################
class AvgTimePerIterColumn(ProgressColumn):
    def render(self, task: TaskID) -> str:  # type: ignore
        if task.completed == 0:  # type: ignore
            return "—"
        avg = task.elapsed / task.completed  # type: ignore
        return f"{avg:.2f}s/it"


class PROGRESS(Progress):
    MAX_TASKS: int = 9
    MIN_TASK_INDEX: Optional[int] = None

    TASK_COLORS = {
        9: "[green]",
        8: "[blue]",
        7: "[magenta]",
        6: "[cyan]",
        5: "[yellow]",
        4: "[red]",
        3: "[white]",
        2: "[bright_green]",
        1: "[bright_blue]",
        0: "[bright_magenta]",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            AvgTimePerIterColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            *args,
            **kwargs,
        )
        self._task_index: TaskID = TaskID(self.MAX_TASKS)

    @classmethod
    def get_active_progress_bar(
        cls, progress: Optional["PROGRESS"] = None
    ) -> "PROGRESS":
        if cls.MIN_TASK_INDEX is not None:
            if isinstance(progress, PROGRESS):
                if (
                    not progress.progress_started()
                    and progress._task_index > cls.MIN_TASK_INDEX
                ):
                    progress.start()
                return progress
            elif progress is None:
                obj = cls.__new__(cls)
                obj.__init__()
                if obj._task_index > cls.MIN_TASK_INDEX:
                    obj.start()
                return obj
        else:
            sig_set = inspect.signature(cls.set_minimum_task_index)
            sig_show = inspect.signature(cls.show)
            sig_hide = inspect.signature(cls.hide)
            msg = (
                "The class variable PROGRESS.MIN_TASK_INDEX is not set."
                f"\nPlease set it to a value between 0 and {cls.MAX_TASKS} using:"
                f"\n\t[bold]PROGRESS.{PROGRESS.set_minimum_task_index.__name__}{sig_set}[/bold]"
                "\nTo show/hide the progress bar, use:"
                f"\n\t[bold]PROGRESS.{PROGRESS.show.__name__}{sig_show}[/bold]"
                f"\n\t[bold]PROGRESS.{PROGRESS.hide.__name__}{sig_hide}[/bold]"
            )
            LOGGER.error(msg)
            exit()

    def soft_start(self) -> None:
        """Only start progress if it was not already active when get_active_progress_bar was called."""
        if not self.progress_started():
            self.start()

    def soft_stop(self) -> None:
        """Only stop progress if there is just one task left."""
        task_index = TaskID(self._task_index + 1)
        if task_index > TaskID(self.MAX_TASKS):
            LOGGER.debug("Last task already removed, skipping")
        else:
            self.remove_task(task_index)
        if task_index == TaskID(self.MAX_TASKS):
            LOGGER.debug("All tasks removed, stopping progress bar")
            self.stop()

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: Optional[float] = 100.0,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> TaskID:
        with self._lock:
            if self._task_index == self.MIN_TASK_INDEX:
                LOGGER.debug(
                    f"Task {description} was not added, due to minimum task index reached."
                )
                return self._task_index  # no more tasks can be added
            fp = fields.pop("fp", None)
            fp_str = format_fp(fp)
            desc = self.TASK_COLORS[int(self._task_index)] + description + fp_str
            task = Task(
                self._task_index,
                desc,
                total,
                completed,
                visible=visible,
                fields=fields,
                _get_time=self.get_time,
                _lock=self._lock,
            )
            self._tasks[self._task_index] = task
            if start:
                self.start_task(self._task_index)
            new_task_index = self._task_index
            self._task_index = TaskID(
                int(self._task_index) - 1
            )  # reverse order of tasks

            # resort tasks
            self._tasks = dict(
                sorted(self._tasks.items())
            )  # ensure tasks are sorted by index
        self.refresh()
        return new_task_index

    def update(self, task_id: TaskID, **kwargs) -> None:
        if task_id in self._tasks.keys():
            super().update(task_id, **kwargs)

    def advance(self, task_id: TaskID, **kwargs) -> None:
        if task_id in self._tasks.keys():
            super().advance(task_id, **kwargs)

    def remove_task(self, task_id: TaskID) -> None:
        """Delete a task if it exists.

        Args:
            task_id (TaskID): A task ID.

        """
        if task_id in self._tasks.keys():
            with self._lock:
                del self._tasks[task_id]
                self._task_index = TaskID(int(self._task_index) + 1)  # increment index

    def progress_started(self) -> bool:
        return any(task.started for task in self._tasks.values())

    def get_description(self, task_id: TaskID) -> str:
        """Get the description of a task."""
        with self._lock:
            return self._tasks[task_id].description if task_id in self._tasks else ""

    @classmethod
    def set_minimum_task_index(cls, index: int) -> None:
        cls.MIN_TASK_INDEX = index

    @classmethod
    def hide(cls) -> None:
        """Turn off the progress bar."""
        cls.set_minimum_task_index(cls.MAX_TASKS)

    @classmethod
    def show(cls) -> None:
        """Turn on the progress bar."""
        cls.set_minimum_task_index(0)


# turn off progress bar if environment variable is set (used for debugging)
if os.environ.get("DISABLE_PROGRESS") == "1":
    PROGRESS.hide()
    LOGGER.debug("Hiding progress bar (DISABLE_PROGRESS environment variable set)")


import logging
from uvicorn.logging import ColourizedFormatter


def get_file_formatter() -> logging.Formatter:
    fmt = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return fmt

def get_stdout_formatter() -> logging.Formatter:
    fmt = ColourizedFormatter(
        "{levelprefix} {message}",
        style="{",
        use_colors=True
    )
    return fmt
import logging
from typing import Literal


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> None:
    handlers = []
    try:
        from rich.console import Console
        from rich.logging import RichHandler

        handlers.append(RichHandler(console=Console(stderr=True), rich_tracebacks=True))
    except ImportError:
        pass

    if not handlers:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
    )

import logging
from typing import Literal, Optional, Union


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    library_loggers: Optional[dict[str, Union[str, int]]] = None,
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

    # Set specific levels for library loggers
    if library_loggers:
        for logger_name, logger_level in library_loggers.items():
            logging.getLogger(logger_name).setLevel(
                logger_level
                if isinstance(logger_level, int)
                else getattr(logging, logger_level)
            )

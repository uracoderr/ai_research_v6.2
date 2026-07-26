"""
Centralized logging configuration.

Why this exists: in the old codebase, agents/*.py called rich's
`console.print("[step]...[/step]")` directly. That looks nice in a
terminal, but the moment the same code runs inside app.py (a web
server), those calls print raw markup/ANSI codes into whatever log
stream the host is capturing - not useful for debugging a deployed
service, and there was no way to filter by log level or module.

Now every module uses Python's standard `logging` instead:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")

And exactly ONE place decides how those log lines are displayed:
- CLI (main.py)  -> setup_logging(pretty=True)  -> colourised Rich output
- Web (app.py)   -> setup_logging(pretty=False) -> plain structured lines,
                     safe for Docker/systemd/log aggregators

Call setup_logging() once, at import time, before any other module in
the app calls get_logger(). Calling it more than once is a harmless no-op.
"""
import logging
import os

_CONFIGURED = False


def setup_logging(pretty: bool = False, level: str = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()  # avoid duplicate lines if uvicorn/rich already attached a handler

    handler = None
    if pretty:
        try:
            from rich.logging import RichHandler
            handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        except ImportError:
            handler = None  # fall through to the plain handler below

    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    root.addHandler(handler)

    # Third-party libraries are noisy at INFO; keep our own logs readable.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

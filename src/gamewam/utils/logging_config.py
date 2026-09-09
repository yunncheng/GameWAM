import logging
from typing import Optional
import os

import torch.distributed as dist

from rich.logging import RichHandler


def setup_logging(
    log_level: int = logging.INFO,
    is_main_process: Optional[bool] = None,
    rich_handler_kwargs: Optional[dict] = None,
    formatter_kwargs: Optional[dict] = None,
    preserve_hydra_handlers: bool = True
) -> None:
    """Configure root logging with Rich formatting.

    In distributed training, only the main process emits logs.
    ``preserve_hydra_handlers`` keeps any existing FileHandlers.
    """
    if is_main_process is None:
        is_main_process = _is_main_process()

    root_logger = logging.getLogger()

    if is_main_process:
        # Save existing FileHandlers (e.g., from Hydra) if requested
        existing_file_handlers = []
        if preserve_hydra_handlers:
            existing_file_handlers = [
                h for h in root_logger.handlers
                if isinstance(h, logging.FileHandler)
            ]

        root_logger.handlers.clear()

        # Configure RichHandler parameters
        default_rich_kwargs = {
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
        if rich_handler_kwargs:
            default_rich_kwargs.update(rich_handler_kwargs)

        # Create RichHandler
        rich_handler = RichHandler(**default_rich_kwargs)

        # Configure Formatter
        default_formatter_kwargs = {
            "fmt": "| >> %(message)s",
            "datefmt": "%m/%d [%H:%M:%S]",
        }
        if formatter_kwargs:
            default_formatter_kwargs.update(formatter_kwargs)
        
        formatter = logging.Formatter(**default_formatter_kwargs)
        rich_handler.setFormatter(formatter)
        
        # Add handler and set logging level
        root_logger.addHandler(rich_handler)

        # Restore existing FileHandlers from Hydra
        for handler in existing_file_handlers:
            root_logger.addHandler(handler)

        root_logger.setLevel(log_level)
        
    else:
        # In non-main processes, set root logger level to ERROR to silence all logs
        root_logger.setLevel(logging.ERROR)


def _is_main_process() -> bool:
    """
    Best-effort check for main process without any synchronization.
    """
    # Prefer torch.distributed state if initialized.
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    # Fallback to environment variables commonly set by launchers.
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")

    return True

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Drop-in replacement for accelerate.logging.get_logger:
    - No implicit barriers.
    - Only the main process emits log records by default.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)


    if not _is_main_process():
        logger.propagate = False
        logger.disabled = True

    return logger

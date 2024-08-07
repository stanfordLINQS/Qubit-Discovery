"""Import local modules"""

import logging
import sys

qd_logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    """Get the Qubit Discovery-wide parent logger."""
    return qd_logger


def log_to_stdout(level: int =logging.INFO) -> logging.Logger:
    """Set the Qubit-Discovery module to log to stdout.

    Parameters
    ----------
        level:
            The minimum level of logs to log.

    Returns
    ----------
        The logger.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler.setFormatter(formatter)
    qd_logger.addHandler(handler)
    qd_logger.setLevel(level)

    return qd_logger

from . import losses
from . import optimization
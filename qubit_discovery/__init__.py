"""Import local modules"""

import logging
import sys

qd_logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    """Get the qubit_discovery-wide parent logger."""
    return qd_logger


def log_to_stdout(level: int =logging.INFO) -> logging.Logger:
    """Set the qubit_discovery package to log to stdout.

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

try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown"
    version_tuple = (0, 0, "unknown")

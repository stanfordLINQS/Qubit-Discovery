"""Import local modules"""

import logging
import sys

from . import losses
from .losses import (
    add_to_metrics,
    build_loss_function,
    describe_metric,
    get_all_metrics,
    ALL_METRICS as metrics
)
from . import optim
from .optim import (
    assign_trunc_nums,
    CircuitSampler,
    run_BFGS,
    run_optimization,
    run_SGD,
    trunc_num_heuristic
)

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    version_tuple = (0, 0, "unknown")


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

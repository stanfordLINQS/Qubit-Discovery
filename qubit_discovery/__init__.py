"""Import local modules"""

import logging
qd_logger = logging.getLogger(__name__)

def get_logger() -> logging.Logger:
    """Get the Qubit Discovery-wide parent logger."""
    return qd_logger

from . import losses
from . import optimization
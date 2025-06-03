import logging
import sys

def setup_logger(name="recsys_logger", level=logging.INFO):
    """Sets up a basic logger."""
    logger = logging.getLogger(name)
    if not logger.handlers: # Avoid adding multiple handlers
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Example usage:
# logger = setup_logger()
# logger.info("This is an info message.")
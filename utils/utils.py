import os
import datetime
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(debug_filename: str = 'debug', console_level: int = logging.INFO) -> None:
    """Attach default Cloud Logging handler to the root logger."""
    logging.root.setLevel(console_level)
    
    # Clear existing handlers
    if logging.root.hasHandlers():
        logging.root.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Add rotating file handler to log to a file
    os.makedirs('logs', exist_ok=True)
    file_handler = RotatingFileHandler(f'logs/{debug_filename}_{timestamp}.log', maxBytes=50*1024*1024, backupCount=5)
    file_handler.setLevel(console_level)
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)

    # Only add console handler if console_level is not DEBUG
    if console_level != logging.DEBUG:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logging.root.addHandler(console_handler)

    logging.info("Local logging setup complete")
    
    # Print current handlers after setup
    print("Handlers after setup:", logging.root.handlers)
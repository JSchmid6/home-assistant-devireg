import logging


def configure_logging(log_filepath: str= None):
    # Configure logging
    logger = logging.getLogger("pysdg")
    logger.setLevel(logging.INFO)

    # Remove all handlers associated with the logger
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    stream_handler = logging.StreamHandler()

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(filename)s:%(lineno)d - %(message)s')
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stream_handler)

    if log_filepath:
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set logging levels for specific libraries
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("pgmpy").setLevel(logging.ERROR)
    logging.getLogger("synthcity").setLevel(logging.ERROR)

    return logger


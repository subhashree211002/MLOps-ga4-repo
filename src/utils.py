import logging

def get_logger(name: str = "pipeline"):
    """
    Configure and return a basic logger.
    """
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO
    )
    return logging.getLogger(name)


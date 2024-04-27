import logging
from utils.functions import process_smiles


def init_global_variable():
    """initialize global variable"""

    global GLB_SMILES
    global GLB_Model
    global GLB_FORWARD_ITERATIONS
    global GLB_DROPOUT_RATE
    global GLB_MAXIMUM_COMPOUNDS
    global LOGGER

    # Configure the software logger
    LOGGER = logging.getLogger(__name__)
    # Set the logging level to INFO
    LOGGER.setLevel(logging.INFO)
    # Create a console handler and set its level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and set the format of log messages
    formatter = logging.Formatter(
        "%(asctime)s - CToxPred2 Logger - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    # Add the console handler to the logger
    LOGGER.addHandler(console_handler)

    # Prediction model settings
    GLB_Model = "rf-ssl"
    GLB_FORWARD_ITERATIONS = 100
    GLB_DROPOUT_RATE = 0.2
    GLB_MAXIMUM_COMPOUNDS = 512
    GLB_SMILES = process_smiles(
        ["O=C1N=C(O)NC1(c1ccccc1)c1ccccc1", "O=C(NC1CC1c2ccccc2)N3CCC(CC3)n4cncn4"]
    )

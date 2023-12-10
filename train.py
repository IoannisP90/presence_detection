from model import create_user_probabilities
import logging


def update_probabilities_file():
    """Function that updates the presence probabilities per user and hour of day.
    Write the probabilities to a csv.
    This is the training script to update the probabilities
    Params: None
    Returns: None
    """
    logging.basicConfig(level=logging.INFO)
    path_flow = "./data/electricity_flow.csv"
    path_temperature = "./data/inside_temperature.csv"

    user_probabilities = create_user_probabilities(path_flow, path_temperature)

    logging.info("Writing user probabilities")
    user_probabilities.to_csv("./model/hourly_user_probabilities.csv", index=False)
    return


update_probabilities_file()

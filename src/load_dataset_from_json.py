import json
import numpy as np

def load_dataset_from_json(json_filename):
    """
    Load dataset (data and labels) from a JSON file.

    Args:
    - json_filename (str): The filename of the JSON file.

    Returns:
    - data (numpy.ndarray): The loaded data.
    - labels (numpy.ndarray): The loaded labels.
    """
    with open(json_filename, "r") as json_file:
        dataset_dict = json.load(json_file)

    # Convert the lists in the dictionary back to NumPy arrays
    data = np.array(dataset_dict["data"])
    labels = np.array(dataset_dict["labels"])

    labels = labels[:, 0].astype(np.int64).reshape(-1, 1)

    return data, labels


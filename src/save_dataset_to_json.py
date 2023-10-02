import json
import numpy as np

def save_dataset_to_json(data, labels, json_filename):
    """
    Save dataset (data and labels) to a JSON file.

    Args:
    - data (numpy.ndarray): The dataset (e.g., MFCC features).
    - labels (numpy.ndarray): The corresponding labels for the dataset.
    - json_filename (str): The filename for the JSON file.

    Returns:
    - None
    """
    dataset_dict = {
        "data": data.tolist(), 
        "labels": labels.tolist()  # Convert NumPy array to a list
    }

    with open(json_filename, "w") as json_file:
        json.dump(dataset_dict, json_file)

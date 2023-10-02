import os

from dataset_load import load_dataset
from save_dataset_to_json import save_dataset_to_json
from load_dataset_from_json import load_dataset_from_json

# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the dataset directory
dataset_dir = os.path.join(script_dir, '..', 'data')

json_filename = "project\data.json"

# need to execute only on the first run

# mfcc, labels = load_dataset(dataset_dir)
# save_dataset_to_json(mfcc, labels, json_filename)

mfcc, labels = load_dataset_from_json(json_filename)
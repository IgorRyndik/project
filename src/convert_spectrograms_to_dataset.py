import os
import numpy as np
from PIL import Image
from save_dataset_to_json import save_dataset_to_json
import gc
import cv2


def create_dataset_from_spectrograms(data_path):
    # List to store image data and labels
    data = []
    labels = []

    # Iterate through the subfolders (speaker IDs)
    for speaker_id in os.listdir(data_path):
        speaker_dir = os.path.join(data_path, speaker_id)
        
        print(speaker_id)

        # Iterate through the png files in each subfolder
        for png_file in os.listdir(speaker_dir):
            # Load the png file using librosa
            png_path = os.path.join(speaker_dir, png_file)
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)
            #img.close()
            
            filename_parts = png_file.split('_')
            digit = int(filename_parts[0])
            labels.append((speaker_id, digit))
        # if int(speaker_id) > 10:
        #     break
        gc.collect()    

    np.save("spectrograms2.npy", data)
    np.save("labels2.npy", labels)

# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, '..', 'spectrograms')
if __name__ == "__main__":
    create_dataset_from_spectrograms(dataset_dir)
    #data = np.load("../spectrograms2.npy")
    #labels = np.load("../labels2.npy")
    #print(data.shape)
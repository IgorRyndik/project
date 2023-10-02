import os
import librosa
import numpy as np

from project.src.feature_extraction import extract_mfcc

def load_dataset():
    # Get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to the dataset directory
    dataset_dir = os.path.join(script_dir, '..', 'data')

    # Initialize empty lists to store data and labels
    data = []
    labels = []

    # Iterate through the subfolders (speaker IDs)
    for speaker_id in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, speaker_id)
        
        print(speaker_id)

        # Iterate through the WAV files in each subfolder
        for wav_file in os.listdir(speaker_dir):
            # Load the WAV file using librosa
            wav_path = os.path.join(speaker_dir, wav_file)
            audio, sr = librosa.load(wav_path, sr=16000)  # Load audio without resampling
            
            # Process the audio (e.g., extract MFCC features)
            mfcc_features = extract_mfcc(audio)
            
            # Extract the digit and recording number from the filename
            filename_parts = wav_file.split('_')
            digit = int(filename_parts[0])
            recording_number = int(filename_parts[2].split('.')[0])
            
            # Append the features and labels to the respective lists
            data.append(mfcc_features)
            labels.append((speaker_id, digit, recording_number))

    #Convert data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

load_dataset()
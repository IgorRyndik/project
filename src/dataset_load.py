import os
import librosa
import numpy as np

from feature_extraction import extract_mfcc
from audio_preprocessing import resample_audio
from save_dataset_to_json import save_dataset_to_json

SAMPLE_RATE = 88200
SEGMENT_LENGTH = 2048
HOP_LENGTH = 1024

def load_dataset(dataset_dir):
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
            audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)  # Load audio without resampling
            
            # Resample audio to achieve the same duration for all recordings
            audio = resample_audio(audio, SAMPLE_RATE)

            # Extract MFCC features for the segment
            mfcc_features = extract_mfcc(audio, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)
                                 
            # Extract the digit and recording number from the filename
            filename_parts = wav_file.split('_')
            digit = int(filename_parts[0])

            # Append the features and labels to the respective lists
            data.append(mfcc_features)
            labels.append((speaker_id, digit))

  
    # Convert data to a NumPy array
    data = np.array(data)

    # Convert labels to a NumPy array (if needed)
    labels = np.array(labels)
    return data, labels

def load_new_users_dataset(dataset_dir, start_label_index:int):
    # Initialize empty lists to store data and labels
    data = []
    labels = []

    users_mapping = {"george": start_label_index, "jackson": start_label_index + 1, 
                     "lucas": start_label_index + 2, "nicolas": start_label_index + 3, 
                     "theo": start_label_index + 4, "yweweler": start_label_index + 5}
    
    # Iterate through the WAV files in each subfolder
    for wav_file in os.listdir(dataset_dir):
        # Extract the digit and recording number from the filename
        filename_parts = wav_file.split('_')
        digit = int(filename_parts[0])
        speaker_name = filename_parts[1]
        speaker_id = users_mapping[speaker_name]

        audio, sr = librosa.load(os.path.join(dataset_dir, wav_file), sr=SAMPLE_RATE)  # Load audio without resampling
            
        # Resample audio to achieve the same duration for all recordings
        audio = resample_audio(audio, SAMPLE_RATE)

            # Extract MFCC features for the segment
        mfcc_features = extract_mfcc(audio, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)

         # Append the features and labels to the respective lists
        data.append(mfcc_features)
        labels.append((speaker_id, digit))

     # Convert data to a NumPy array
    data = np.array(data)

    # Convert labels to a NumPy array (if needed)
    labels = np.array(labels)
    return data, labels


if __name__ == "__main__":
    json_filename = "data_sr_88.2k.json"
    # Get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to the dataset directory
    dataset_dir = os.path.join(script_dir, '..', 'data')
    mfcc, labels = load_dataset(dataset_dir)
    save_dataset_to_json(mfcc, labels, json_filename)
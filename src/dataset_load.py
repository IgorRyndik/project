import os
import librosa
import numpy as np

from feature_extraction import extract_mfcc
from audio_preprocessing import resample_audio 

SAMPLE_RATE = 8000
SEGMENT_LENGTH = 256
HOP_LENGTH = 128

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
            audio = resample_audio(audio)

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

    
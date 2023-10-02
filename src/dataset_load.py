import os
import librosa
import numpy as np

from feature_extraction import extract_mfcc

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 1024
HOP_LENGTH = 256

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
            audio, sr = librosa.load(wav_path, sr=16000)  # Load audio without resampling
            
            # Calculate the number of segments to divide the audio into
            num_segments = len(audio) // SEGMENT_LENGTH

            # Extract MFCC features for each segment
            for i in range(num_segments):
                start = i * HOP_LENGTH
                end = start + SEGMENT_LENGTH
                segment = audio[start:end]
                
                # Extract MFCC features for the segment
                mfcc_features = extract_mfcc(segment, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)
                
                 
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

    
import os
import json
import keras
import time
import tensorflow as tf
import numpy as np
import librosa

from feature_extraction import extract_mfcc
from audio_preprocessing import resample_audio

from prepare_dataset import prepare_datasets
from plot_history import plot_history

SEGMENT_LENGTH = 256
HOP_LENGTH = 128

def pretrained_recordings_test():
    # Start measuring time
    start_time = time.time()

    # Define the relative path to the models folder
    recordings_folder = os.path.join(script_dir, '..', 'recordings')

    audio, sr = librosa.load(os.path.join(recordings_folder, '1_jackson_5.wav'), sr=8000)  # Load audio without resampling
                
    # Resample audio to achieve the same duration for all recordings
    audio = resample_audio(audio, 8000)
    data = []

    # Extract MFCC features for the segment
    mfcc_features = extract_mfcc(audio, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)
    data.append(mfcc_features)

    data = np.array(data)
    # add an axis to input sets
    data = data[..., np.newaxis]
    labels = np.array([0])

    # evaluate model on test set
    prediction = pretrained_model.predict(data)
    print('\nPrediction:', np.argmax(prediction)==1)

    # Stop measuring time
    end_time = time.time()

    # Calculate the training time in seconds
    training_time_seconds = end_time - start_time

    # Convert training time to hours, minutes, and seconds
    training_hours, remainder = divmod(training_time_seconds, 3600)
    training_minutes, training_seconds = divmod(remainder, 60)

    # Print the training time in a human-readable format
    print(f"Training time: {int(training_hours):02d} hours, {int(training_minutes):02d} minutes, {int(training_seconds):02d} seconds")

    # Start measuring time
    start_time = time.time()

    # Define the relative path to the models folder
    recordings_folder = os.path.join(script_dir, '..', 'recordings')

    audio, sr = librosa.load(os.path.join(recordings_folder, '1_jackson_4.wav'), sr=8000)  # Load audio without resampling
                
    # Resample audio to achieve the same duration for all recordings
    audio = resample_audio(audio, 8000)
    data = []
    labels = []

    # Extract MFCC features for the segment
    mfcc_features = extract_mfcc(audio, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)
    data.append(mfcc_features)

    data = np.array(data)
    # add an axis to input sets
    data = data[..., np.newaxis]
    labels = np.array([0])

    # evaluate model on test set
    prediction = pretrained_model.predict(data)
    print('\nPrediction:', np.argmax(prediction)==1)

    # Stop measuring time
    end_time = time.time()


    # Calculate the execution time in milliseconds
    execution_time_ms = (end_time - start_time) * 1000

    # Print or use the execution time as needed
    print(f"Execution time: {execution_time_ms:.2f} ms")



# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the models folder
models_folder = os.path.join(script_dir, '..', 'models')
# Load the pre-trained model
pretrained_model = tf.keras.models.load_model(os.path.join(models_folder, 'mfcc_digits_model_4l_60e_04d_2.h5'))



# Start measuring time
start_time = time.time()

    # Define the relative path to the models folder
myvoice_folder = os.path.join(script_dir, '..', 'myvoice')

audio, sr = librosa.load(os.path.join(myvoice_folder, 'me_1_2.wav'), sr=8000)  # Load audio without resampling
                
    # Resample audio to achieve the same duration for all recordings
audio = resample_audio(audio, 8000)
data = []

# Extract MFCC features for the segment
mfcc_features = extract_mfcc(audio, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)
data.append(mfcc_features)

data = np.array(data)
# add an axis to input sets
data = data[..., np.newaxis]
labels = np.array([0])

# evaluate model on test set
prediction = pretrained_model.predict(data)
print('\nPrediction:', np.argmax(prediction)==1)

# Stop measuring time
end_time = time.time()

# Calculate the training time in seconds
training_time_seconds = end_time - start_time

# Convert training time to hours, minutes, and seconds
training_hours, remainder = divmod(training_time_seconds, 3600)
training_minutes, training_seconds = divmod(remainder, 60)

# Print the training time in a human-readable format
print(f"Training time: {int(training_hours):02d} hours, {int(training_minutes):02d} minutes, {int(training_seconds):02d} seconds")

# Start measuring time
start_time = time.time()

audio, sr = librosa.load(os.path.join(myvoice_folder, 'me_0_1.wav'), sr=8000)  # Load audio without resampling
                
# Resample audio to achieve the same duration for all recordings
audio = resample_audio(audio, 8000)
data = []
labels = []

# Extract MFCC features for the segment
mfcc_features = extract_mfcc(audio, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)
data.append(mfcc_features)

data = np.array(data)
# add an axis to input sets
data = data[..., np.newaxis]
labels = np.array([0])

# evaluate model on test set
prediction = pretrained_model.predict(data)
print('\nPrediction:', np.argmax(prediction)==0)

# Stop measuring time
end_time = time.time()


# Calculate the execution time in milliseconds
execution_time_ms = (end_time - start_time) * 1000

# Print or use the execution time as needed
print(f"Execution time: {execution_time_ms:.2f} ms")
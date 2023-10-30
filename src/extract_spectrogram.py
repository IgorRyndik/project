import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
from audio_preprocessing import resample_audio
from save_dataset_to_json import save_dataset_to_json

def create_spectrograms_data(dataset_dir):
    # data = []
    # labels = []

    width = 192  / 80  # 80 pixels per inch
    height = 315 / 80  # 80 pixels per inch

     # Iterate through the subfolders (speaker IDs)
    for speaker_id in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, speaker_id)
        
        print(speaker_id)

        # Iterate through the WAV files in each subfolder
        for wav_file in os.listdir(speaker_dir):
            # Load the WAV file using librosa
            wav_path = os.path.join(speaker_dir, wav_file)
            audio, sr = librosa.load(wav_path, sr=None, mono=True)  # Load audio without resampling

            # Resample audio to achieve the same duration for all recordings
            audio = resample_audio(audio)

            # Create the spectrogram
            spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            # Display the spectrogram
            plt.figure(figsize=(height, width))  # Set the size of the spectrogram plot (192x315 pixels)
            librosa.display.specshow(spectrogram, sr=sr, cmap='viridis')
            # Remove axis labels and color bar
            plt.axis('off')

            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            save_dir_path = os.path.join(script_dir, '..', 'spectrograms', speaker_id)
            if not os.path.exists(save_dir_path):
                os.mkdir(save_dir_path  )

            save_file_path = os.path.join(save_dir_path, f"{wav_file}.png")
           
            # Save the spectrogram as an image
            plt.savefig(save_file_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Add a `channels` dimension, so that the spectrogram can be used
            # as image-like input data with convolution layers (which expect
            # shape (`batch_size`, `height`, `width`, `channels`).
            #spectrogram = spectrogram[..., tf.newaxis]

            # Extract the digit and recording number from the filename
            # filename_parts = wav_file.split('_')
            # digit = int(filename_parts[0])

            # data.append(spectrogram)
            # labels.append((speaker_id, digit))
    
    # print("Convertring data into numpy array")
    # # Convert data to a NumPy array
    # data = np.array(data)

    # print("Converting labels into numpy array")
    # # Convert labels to a NumPy array (if needed)
    # labels = np.array(labels)
    #return data, labels
            


# Get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, '..', 'data')
if __name__ == "__main__":
    create_spectrograms_data(dataset_dir)
    # data, labels = create_spectrograms_data(dataset_dir)
    # np.save("spectrograms.npy", data)
    # np.save("labels.npy", labels)
    #save_dataset_to_json(data, labels, 'data_spectrograms.json')
import os
import tensorflow as tf
import librosa

from pydub import AudioSegment
from audio_preprocessing import resample_audio
from feature_extraction import extract_mfcc
from pydub.silence import split_on_silence
import numpy as np

SEGMENT_LENGTH = 256
HOP_LENGTH = 128

def convert_voice_to_pass(sound_file='me_multiple_digits.wav'):

    # Get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the models folder
    models_folder = os.path.join(script_dir, '..', 'models')
    # Load the pre-trained model
    pretrained_model = tf.keras.models.load_model(os.path.join(models_folder, 'mfcc_digits_model_4l_60e_04d_2.h5'))

    # Define the relative path to the models folder
    recordings_folder = os.path.join(script_dir, '..', 'myvoice')
    sound = AudioSegment.from_wav(os.path.join(recordings_folder, sound_file))
    dBFS = sound.dBFS

    chunks = split_on_silence(sound, 
        min_silence_len = 100,                         # minimum length of silence:250 ms 
        silence_thresh = dBFS-16,                      # threshhold to divide voice and silence
        keep_silence = 50                             # time left before and after each voice cut:50 ms
        )

    password = ''
    for j, chunk in enumerate(chunks):
        chunk.export(f'me_digits{j}.wav', bitrate = "192k", format = "wav")
        audio, sr = librosa.load(f'me_digits{j}.wav', sr=8000)  # Load audio without resampling
        os.remove(f'me_digits{j}.wav')
        # Resample audio to achieve the same duration for all recordings
        audio = resample_audio(audio, 8000)
        data = []

        # Extract MFCC features for the segment
        mfcc_features = extract_mfcc(audio, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)
        data.append(mfcc_features)

        data = np.array(data)
        # add an axis to input sets
        data = data[..., np.newaxis]

        # evaluate model on test set
        prediction = pretrained_model.predict(data, verbose=0)
        password += str(np.argmax(prediction))

    return password

if __name__ == '__main__':
    password = convert_voice_to_pass()
    print(f'Password: {password}')


import os
import tensorflow as tf
import librosa
import time
from pydub import AudioSegment
from audio_preprocessing import resample_audio
from feature_extraction import extract_mfcc
from pydub.silence import split_on_silence
from feature_extraction import extract_mfcc
from audio_preprocessing import resample_audio
from users_database import check_user_password
import numpy as np

SEGMENT_LENGTH = 256
HOP_LENGTH = 128

def authorize_by_voice_and_password(voice_input='45_0123.wav', user_id=45):
    # Get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the models folder
    models_folder = os.path.join(script_dir, '..', 'models')
    
    # Load the pre-trained model
    digits_pretrained_model = tf.keras.models.load_model(os.path.join(models_folder, 'mfcc_digits_model_4l_60e_04d_2.h5'))

    # Load the pre-trained model
    speech_pretrained_model = tf.keras.models.load_model(os.path.join(models_folder, 'mfcc_model_4l_60e_04d_retrained_2.h5'))

    # Define the relative path to the models folder
    recordings_folder = os.path.join(script_dir, '..', 'speaker_recognition_test')

    audio, sr = librosa.load(os.path.join(recordings_folder, voice_input), sr=8000)  # Load audio without resampling
    audio = resample_audio(audio, 8000)
    # Extract MFCC features for the segment
    data = []
    mfcc_features = extract_mfcc(audio, sr = sr, n_fft=SEGMENT_LENGTH, hop_length=HOP_LENGTH)
    data.append(mfcc_features)

    data = np.array(data)
    # add an axis to input sets
    data = data[..., np.newaxis]

    prediction = speech_pretrained_model.predict(data)
    print(np.argmax(prediction))

    prediction = digits_pretrained_model.predict(data)
    print(np.argmax(prediction))

    # Start measuring time
    start_time = time.time()

    sound = AudioSegment.from_wav(os.path.join(recordings_folder, voice_input))
    dBFS = sound.dBFS

    chunks = split_on_silence(sound, 
        min_silence_len = 120,                         # minimum length of silence:250 ms 
        silence_thresh = dBFS-16,                     # threshhold to divide voice and silence
        keep_silence = 120                             # time left before and after each voice cut:50 ms
        )
    
    password = ''
    user_id_prediciton = []
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
        digit_prediction = digits_pretrained_model.predict(data, verbose=0)
        password += str(np.argmax(digit_prediction))
        # evaluate model on test set
        prediction = speech_pretrained_model.predict(data)
        user_id_prediciton.append(np.argmax(prediction))
    
    # Stop measuring time
    end_time = time.time()

    # Calculate the execution time in milliseconds
    execution_time_ms = (end_time - start_time) * 1000

    # Print or use the execution time as needed
    print(f"Execution time: {execution_time_ms:.2f} ms")
    return password, user_id_prediciton

if __name__ == '__main__':
    voice_input='6_9876.wav'
    user_id=6
    password, user_ids = authorize_by_voice_and_password(voice_input, user_id)
    print(f'Password: {password}')
    print(f'Voice verified: { user_ids} {all(item == user_id for item in user_ids)}')
    print(f'Password verified: {check_user_password(user_id, password)}')


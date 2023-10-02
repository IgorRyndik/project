import librosa
import numpy as np

def extract_mfcc(audio, sr=16000, n_mfcc=13, n_fft=1024, hop_length=256):
    """
    Extract MFCC features from audio data using librosa.

    Args:
    - audio (numpy.ndarray): The audio waveform.
    - sr (int): The sample rate of the audio (default: 16000 Hz).
    - n_mfcc (int): The number of MFCC coefficients to extract (default: 13).
    - n_fft (int): The length of the FFT window (default: 1024 samples).
    - hop_length (int): The number of samples between successive frames (default: 256 samples).

    Returns:
    - mfcc_features (numpy.ndarray): The extracted MFCC features.
    """
    # Compute MFCCs
    mfcc_features = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    # Transpose the feature matrix for the desired shape (time steps x MFCC coefficients)
    mfcc_features = mfcc_features.T

    return mfcc_features

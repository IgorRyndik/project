from scipy.signal import resample

def resample_audio(audio, target_sample_rate = 8000):
    return resample(audio, target_sample_rate)
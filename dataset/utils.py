# utils.py
import librosa
import numpy as np

def load_audio(file_path, sr=22050, mono=True):
    """
    Loads an audio file using librosa.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        mono (bool): Whether to convert the audio signal to mono.
    
    Returns:
        np.ndarray: Audio time series.
        int: Sampling rate of the audio file.
    """
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=mono)
    return audio, sample_rate

def compute_spectrogram(audio, n_fft=1024, hop_length=512):
    """
    Computes the magnitude spectrogram of an audio signal using STFT.
    
    Args:
        audio (np.ndarray): Audio time series.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive frames.
    
    Returns:
        np.ndarray: Magnitude spectrogram.
    """
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(spectrogram)
    return magnitude

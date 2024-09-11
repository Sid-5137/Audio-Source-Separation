# utils.py
import librosa
import numpy as np
import torch
import soundfile as sf

def load_audio(file_path, sr=22050, mono=True):
    """
    Loads an audio file using Librosa with error handling.
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate to load the audio.
        mono (bool): Convert audio to mono if True.
    Returns:
        np.ndarray: Loaded audio signal.
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=mono)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(1)  # Return silent signal on error

def compute_spectrogram(audio, n_fft=1024, hop_length=512):
    """
    Computes and returns the magnitude spectrogram of the audio signal.
    Args:
        audio (np.ndarray): Input audio signal.
        n_fft (int): FFT size.
        hop_length (int): Number of samples between frames.
    Returns:
        np.ndarray: Magnitude spectrogram.
    """
    spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    return spectrogram / (np.max(spectrogram) + 1e-6)  # Normalize to avoid division by zero

def write_wav(path, audio, sr):
    """
    Writes audio data to a WAV file using soundfile.
    Args:
        path (str): Output file path.
        audio (np.ndarray): Audio data.
        sr (int): Sampling rate.
    """
    sf.write(path, audio.T, sr, format='PCM_16')

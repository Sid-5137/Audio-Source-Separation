# dataset.py
import os
import librosa
import numpy as np
from torch.utils.data import Dataset
import torch

class MUSDB18HQDataset(Dataset):
    def __init__(self, root_dir, n_fft=1024, hop_length=512, transform=None):
        self.root_dir = root_dir
        self.tracks = os.listdir(root_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_dir = os.path.join(self.root_dir, self.tracks[idx])
        mixture, sr = librosa.load(os.path.join(track_dir, 'mixture.wav'), sr=None)
        vocals, _ = librosa.load(os.path.join(track_dir, 'vocals.wav'), sr=None)
        drums, _ = librosa.load(os.path.join(track_dir, 'drums.wav'), sr=None)
        bass, _ = librosa.load(os.path.join(track_dir, 'bass.wav'), sr=None)
        other, _ = librosa.load(os.path.join(track_dir, 'other.wav'), sr=None)

        mixture_spec = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        vocals_spec = librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length)
        drums_spec = librosa.stft(drums, n_fft=self.n_fft, hop_length=self.hop_length)
        bass_spec = librosa.stft(bass, n_fft=self.n_fft, hop_length=self.hop_length)
        other_spec = librosa.stft(other, n_fft=self.n_fft, hop_length=self.hop_length)

        mixture_mag = np.abs(mixture_spec)
        vocals_mag = np.abs(vocals_spec)
        drums_mag = np.abs(drums_spec)
        bass_mag = np.abs(bass_spec)
        other_mag = np.abs(other_spec)

        targets_mag = np.stack([vocals_mag, drums_mag, bass_mag, other_mag], axis=0)

        if self.transform:
            mixture_mag = self.transform(mixture_mag)
            targets_mag = self.transform(targets_mag)

        # Convert to tensors
        mixture_mag = torch.tensor(mixture_mag, dtype=torch.float32)
        targets_mag = torch.tensor(targets_mag, dtype=torch.float32)

        return mixture_mag, targets_mag, mixture_spec, vocals_spec, drums_spec, bass_spec, other_spec

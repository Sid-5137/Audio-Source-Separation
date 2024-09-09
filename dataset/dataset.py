# dataset.py
import torch
from torch.utils.data import Dataset
import torchaudio
import os

# Custom Dataset for MUSDB18-HQ using STFT
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
        mixture, sr = torchaudio.load(os.path.join(track_dir, 'mixture.wav'))
        vocals, _ = torchaudio.load(os.path.join(track_dir, 'vocals.wav'))
        drums, _ = torchaudio.load(os.path.join(track_dir, 'drums.wav'))
        bass, _ = torchaudio.load(os.path.join(track_dir, 'bass.wav'))
        other, _ = torchaudio.load(os.path.join(track_dir, 'other.wav'))

        mixture_spec = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        vocals_spec = torch.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        drums_spec = torch.stft(drums, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        bass_spec = torch.stft(bass, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        other_spec = torch.stft(other, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)

        mixture_mag = torch.abs(mixture_spec)
        targets_mag = torch.stack([torch.abs(vocals_spec), torch.abs(drums_spec), torch.abs(bass_spec), torch.abs(other_spec)], dim=0)

        if self.transform:
            mixture_mag = self.transform(mixture_mag)
            targets_mag = self.transform(targets_mag)

        return mixture_mag, targets_mag, mixture_spec, vocals_spec, drums_spec, bass_spec, other_spec


import h5py
import torch
from torch.utils.data import Dataset

class MUSDBDataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.hdf5_file = hdf5_file
        self.transform = transform

        # Open the HDF5 file in read mode
        self.h5f = h5py.File(self.hdf5_file, 'r')

        # Collect all track keys from the HDF5 file
        self.track_keys = list(self.h5f.keys())

    def __len__(self):
        # Return the total number of tracks in the dataset
        return len(self.track_keys)

    def __getitem__(self, idx):
        # Get the track key for the given index
        track_key = self.track_keys[idx]
        
        # Load log-Mel spectrograms (assuming they are stored in HDF5 as 2D arrays)
        mixture = self.h5f[track_key]['mixture_log_mel'][:]
        vocals = self.h5f[track_key]['vocals_log_mel'][:]
        drums = self.h5f[track_key]['drums_log_mel'][:]
        bass = self.h5f[track_key]['bass_log_mel'][:]
        other = self.h5f[track_key]['other_log_mel'][:]

        # Convert them to 3D tensors (height, width) -> (1, height, width)
        mixture = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        vocals = torch.tensor(vocals, dtype=torch.float32).unsqueeze(0)
        drums = torch.tensor(drums, dtype=torch.float32).unsqueeze(0)
        bass = torch.tensor(bass, dtype=torch.float32).unsqueeze(0)
        other = torch.tensor(other, dtype=torch.float32).unsqueeze(0)

        # Stack the stems into a single 4D tensor [channels, height, width]
        stems = torch.stack([vocals, drums, bass, other], dim=0)

        # Apply any transformations if needed
        if self.transform:
            mixture = self.transform(mixture)
            stems = self.transform(stems)

        return mixture, stems

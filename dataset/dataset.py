# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm
from utils import load_audio, compute_spectrogram

class MUSDB18HQDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes, random_hops, hdf_dir, audio_transform=None, in_memory=False):
        """
        Initializes a source separation dataset using HDF5 for efficient storage and retrieval.
        :param dataset: Dictionary containing paths to audio files for training/validation.
        :param partition: Data partition (train/val).
        :param instruments: List of instruments (e.g., ['vocals', 'drums', 'bass', 'other']).
        :param sr: Sampling rate for audio files.
        :param channels: Number of audio channels (1 for mono, 2 for stereo).
        :param shapes: Dictionary specifying the shapes of input/output frames.
        :param random_hops: Boolean flag to randomly sample positions in the audio.
        :param hdf_dir: Directory where HDF5 files are stored.
        :param audio_transform: Optional audio transformation function.
        :param in_memory: Boolean flag to load the entire dataset into memory.
        """
        super(MUSDB18HQDataset, self).__init__()
        self.hdf_dataset = None
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments

        # Prepare HDF5 file
        self._prepare_hdf5(dataset, partition)

        # Load lengths for setting sampling positions
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(idx)].attrs["target_length"] for idx in range(len(f))]
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]
        self.start_pos = np.cumsum(lengths)
        self.length = self.start_pos[-1]

    def _prepare_hdf5(self, dataset, partition):
        """
        Prepares the HDF5 file by loading and processing the audio data if not already done.
        """
        if not os.path.exists(self.hdf_dir):
            os.makedirs(os.path.dirname(self.hdf_dir), exist_ok=True)
            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = self.sr
                f.attrs["channels"] = self.channels
                f.attrs["instruments"] = self.instruments

                for idx, example in enumerate(tqdm(dataset[partition])):
                    mix_audio, _ = load_audio(example["mix"], sr=self.sr, mono=(self.channels == 1))
                    source_audios = [load_audio(example[source], sr=self.sr, mono=(self.channels == 1))[0] for source in self.instruments]
                    source_audios = np.concatenate(source_audios, axis=0)
                    
                    assert source_audios.shape[1] == mix_audio.shape[1], "Source and mix audio lengths do not match."

                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                    grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                    grp.attrs["length"] = mix_audio.shape[1]
                    grp.attrs["target_length"] = source_audios.shape[1]

    def __getitem__(self, index):
        if self.hdf_dataset is None:
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver="core" if self.in_memory else None)

        audio_idx = np.searchsorted(self.start_pos, index)
        if audio_idx > 0:
            index -= self.start_pos[audio_idx - 1]

        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        target_length = self.hdf_dataset[str(audio_idx)].attrs["target_length"]

        start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1)) if self.random_hops else index * self.shapes["output_frames"]
        start_pos = max(0, start_target_pos - self.shapes["output_start_frame"])
        end_pos = min(audio_length, start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"])

        pad_front = max(0, -start_pos)
        pad_back = max(0, end_pos - audio_length)

        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant")

        targets = self.hdf_dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant")

        targets = {inst: targets[idx*self.channels:(idx+1)*self.channels] for idx, inst in enumerate(self.instruments)}

        if self.audio_transform:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

    def __len__(self):
        return self.length

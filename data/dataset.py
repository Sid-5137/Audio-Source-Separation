import os
import torch
from torch.utils.data import Dataset
import musdb
import torchaudio
from tqdm import tqdm

def preprocess_dataset(musdb_root, output_dir="musdb_specs", subset="train", limit=None):
    os.makedirs(output_dir, exist_ok=True)
    mus = musdb.DB(root=musdb_root, is_wav=True)
    tracks = mus.load_mus_tracks(subsets=subset)
    if limit is not None:
        tracks = tracks[:limit]
    print(f"Processing {len(tracks)} tracks")
    transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=2)
    for i, track in enumerate(tqdm(tracks, desc=f"Preprocessing {subset} Tracks")):
        mix = torch.tensor(track.audio.T, dtype=torch.float32)
        targets = [torch.tensor(track.targets[src].audio.T, dtype=torch.float32) 
                   for src in ['vocals', 'drums', 'bass', 'other']]
        mix_spec = torch.log1p(transform(mix))
        target_spec = torch.stack([torch.log1p(transform(t)) for t in targets])
        mix_chunks = chunk_spectrogram(mix_spec, 512)
        target_chunks = chunk_spectrogram(target_spec, 512)
        print(f"Track {i}: {mix_spec.shape[-1]} frames, {len(mix_chunks)} mix chunks, {len(target_chunks)} target chunks")
        torch.save({'spec': mix_spec, 'chunks': mix_chunks}, f"{output_dir}/mix_{i}.pt")
        torch.save({'spec': target_spec, 'chunks': target_chunks}, f"{output_dir}/target_{i}.pt")
    return len(tracks)

def chunk_spectrogram(spec, chunk_size=512):
    num_frames = spec.shape[-1]
    chunks = []
    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        chunk = spec[..., start:end]
        if chunk.shape[-1] < chunk_size:
            padding = (0, chunk_size - chunk.shape[-1])
            chunk = torch.nn.functional.pad(chunk, padding)
        chunks.append(chunk)
    return chunks

class PrecomputedMusdbDataset(Dataset):
    def __init__(self, spec_dir="musdb_specs", chunk_size=512):
        self.spec_dir = spec_dir
        self.chunk_size = chunk_size
        self.chunk_indices = self._prepare_chunk_indices()

    def _prepare_chunk_indices(self):
        chunk_indices = []
        i = 0
        while os.path.exists(f"{self.spec_dir}/mix_{i}.pt"):
            data = torch.load(f"{self.spec_dir}/mix_{i}.pt")
            num_chunks = len(data['chunks'])
            for j in range(num_chunks):
                chunk_indices.append((i, j))
            i += 1
        return chunk_indices

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        track_idx, chunk_idx = self.chunk_indices[idx]
        mix_data = torch.load(f"{self.spec_dir}/mix_{track_idx}.pt")
        target_data = torch.load(f"{self.spec_dir}/target_{track_idx}.pt")
        return mix_data['chunks'][chunk_idx], target_data['chunks'][chunk_idx]
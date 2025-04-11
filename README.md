# Custom U-Net for Audio Source Separation

This project implements a custom U-Net model for audio source separation, targeting the MUSDB18-HQ dataset. The model separates stereo audio mixtures into four stems: vocals, drums, bass, and "other" (remaining instruments), using a spectrogram-based approach with PyTorch.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## Overview

The U-Net model is designed to predict log-magnitude spectrograms for four stereo sources from a mixed audio input. It leverages a custom `StereoSpectrogramLoss` combining log-magnitude and linear-magnitude terms. The implementation uses precomputed spectrograms to optimize training speed and supports multi-GPU training with `torch.nn.DataParallel`. This project was developed with flexibility for both local machines (e.g., GTX 1650) and high-performance servers (e.g., dual GPUs with 256 GB RAM).

## Dataset

- **MUSDB18-HQ**: A high-quality dataset with 150 music tracks (100 train, 50 test) at 44.1 kHz, including separate stems for mixture, vocals, drums, bass, and "other."
- **Preprocessing**: Audio is converted to log-magnitude spectrograms (`n_fft=1024`, `hop_length=256`) and chunked into 512-frame segments, saved as `.pt` files (~50 GB for training set).
- **Download**: Obtain MUSDB18-HQ from [its official repository](https://sigsep.github.io/datasets/musdb.html) and place it in `./musdb18hq`.

## Requirements

- Python 3.11 or later
- PyTorch (with CUDA for GPU support)
- torchaudio
- musdb
- soundfile
- tqdm
- Optional: mir_eval (for SDR evaluation), matplotlib (for visualization)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Sid-5137/Audio-Source-Separation.git
   cd Audio-Source-Separation
   ```

2. Install dependencies:

   ```bash
   pip install torch torchaudio musdb soundfile tqdm
   # Optional: pip install mir_eval matplotlib
   ```

3. Download and extract MUSDB18-HQ into `./musdb18hq`.

## Usage

The project is modular, with `main.py` as the entry point. Use command-line arguments to preprocess, train, or test the model.

- **Preprocess Dataset**:
  ```bash
  python main.py --mode preprocess --musdb_root ./musdb18hq --spec_dir musdb_specs
  ```
  Generates ~7,754 chunks (~50 GB) in `musdb_specs/`.

- **Train**:
  ```bash
  python main.py --mode train --spec_dir musdb_specs --epochs 10 --batch_size 32
  ```
  Saves model weights as `unet_model_epochX.pth`.

- **Test**:
  ```bash
  python main.py --mode test --audio_path /path/to/audio.wav --batch_size 32
  ```
  Outputs separated WAVs in `separated_audio/`.

## Training

- **Model**: `ModifiedUNet` (placeholder; replace with your full U-Net architecture).
- **Loss**: `StereoSpectrogramLoss` (weighted sum of log-magnitude and linear-magnitude MSE).
- **Optimizer**: Adam (lr=0.0001).
- **Hardware**:
  - Local: GTX 1650 (4 GB VRAM), batch_size=2, ~3 hours/epoch (pre-optimization).
  - Server: Dual GPUs, 256 GB RAM, batch_size=32, ~24 seconds/epoch (estimated).
- **Setup**: Precomputed spectrograms, multi-GPU support via `DataParallel`, `num_workers=16` for data loading.

Run training:
```bash
python main.py --mode train --spec_dir musdb_specs --epochs 10 --batch_size 32
```

## Testing

Test on a single stereo WAV file:
```bash
python main.py --mode test --audio_path ./musdb18hq/test/Al\ James\ -\ Schoolboy\ Facination/mixture.wav --batch_size 32
```
- Outputs: `mixture_mix.wav`, `mixture_vocals.wav`, `mixture_drums.wav`, `mixture_bass.wav`, `mixture_other.wav`.
- Phase: Uses mix’s phase for reconstruction (magnitude-only prediction).

## Results

- **Epoch 1**: Loss = 0.0474 (3+ hours on GTX 1650, pre-optimization).
  - Decent separation for vocals and drums; some bleed in bass and "other."
- **Optimized**: Precomputed spectrograms reduced epoch time to ~7–10 minutes locally; server expected at ~4–5 minutes for 10 epochs.
- **Evaluation**: SDR not yet implemented (requires `mir_eval` integration).

## Troubleshooting

- **Memory Errors**: Reduce `batch_size` (e.g., 16 or 8) if VRAM < 20 GB/GPU.
- **Slow Training**: Ensure `musdb_specs/` is on an SSD; adjust `num_workers` to match CPU cores.
- **Size Mismatch**: Verify chunking and trimming align spectrogram dimensions (fixed in latest code).
- **Multiprocessing Issues**: Keep `if __name__ == "__main__":` guards for server runs.

## Acknowledgements

- MUSDB18-HQ creators for the dataset.
- PyTorch and torchaudio communities for robust tools.
- xAI’s Grok for assistance in debugging and optimization.

*Last Updated: April 10, 2025*


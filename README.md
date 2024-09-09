# Custom U-Net for Source Separation

This project implements a custom U-Net model for audio source separation, specifically targeting the MUSDB18-HQ dataset. The model separates an audio mixture into four stems: vocals, drums, bass, and other.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Validation](#validation)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## Overview

The U-Net model is designed to separate audio mixtures into their component sources by learning masks applied to spectrograms. This implementation leverages `librosa` for audio processing, PyTorch for model training, and `mir_eval` for evaluation metrics like Signal-to-Distortion Ratio (SDR).

## Dataset

- **MUSDB18-HQ**: This dataset contains high-quality recordings of various music tracks with individual stems (vocals, drums, bass, and other).
- Download the MUSDB18-HQ dataset from the official website or repository.
- Organize the dataset into `train`, `valid`, and `test` directories with each track containing separate `.wav` files for mixture, vocals, drums, bass, and other.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.11 or later
- PyTorch
- torchaudio
- librosa
- mir_eval
- tqdm
- matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Sid-5137/Audio-Source-Separation.git
   cd Audio-Source-Separation

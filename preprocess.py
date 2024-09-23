import os
import librosa
import numpy as np
import h5py
from tqdm import tqdm

# Paths for the MUSDB18-HQ dataset
train_dir = '/media/sid/Sid-HDD/archive/train'
test_dir = '/media/sid/Sid-HDD/archive/test'
valid_dir = '/media/sid/Sid-HDD/archive/valid'

# Paths for HDF5 files
output_dir = "/home/sid/Desktop/Work/Samsung_Prism/new_sep/data"
os.makedirs(output_dir, exist_ok=True)
train_hdf5_path = os.path.join(output_dir, "musdb18_train_30sec.h5")
test_hdf5_path = os.path.join(output_dir, "musdb18_test_30sec.h5")
valid_hdf5_path = os.path.join(output_dir, "musdb18_valid_30sec.h5")

# Set fixed length for audio (30 seconds)
target_length_sec = 30  # 30 seconds of audio
target_sr = 44100  # Resample to 44.1kHz
target_length_samples = target_length_sec * target_sr
n_mels = 128  # Number of Mel bands
fmin = 0  # Minimum frequency
fmax = None  # Maximum frequency (default is Nyquist frequency)

# Function to select the middle 30 seconds of the track
def select_best_segment(audio, sr, target_length_samples):
    total_samples = audio.shape[0]
    
    # If the audio is shorter than the target, pad it with zeros
    if total_samples <= target_length_samples:
        return librosa.util.fix_length(audio, target_length_samples)
    
    # Otherwise, select the middle 30 seconds
    start_sample = (total_samples - target_length_samples) // 2
    end_sample = start_sample + target_length_samples
    return audio[start_sample:end_sample]

# Function to compute Log-Mel spectrogram
def compute_log_mel_spectrogram(audio, sr, n_mels=128, fmin=0, fmax=None):
    # Compute Mel spectrogram with audio as a keyword argument
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)
    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

# Function to process a split and save to HDF5
def preprocess_split(data_dir, output_hdf5_path):
    with h5py.File(output_hdf5_path, 'w') as hdf5_file:
        # Loop through each track subfolder
        for track_name in tqdm(os.listdir(data_dir)):
            track_path = os.path.join(data_dir, track_name)
            if os.path.isdir(track_path):
                print(f"Processing track: {track_name}")
                # Create a group in the HDF5 file for this track
                track_group = hdf5_file.create_group(track_name)
                
                # Paths to the stem files
                stem_files = {
                    "mixture": os.path.join(track_path, "mixture.wav"),
                    "vocals": os.path.join(track_path, "vocals.wav"),
                    "drums": os.path.join(track_path, "drums.wav"),
                    "bass": os.path.join(track_path, "bass.wav"),
                    "other": os.path.join(track_path, "other.wav")
                }

                # Process each stem and save to HDF5
                for stem, file_path in stem_files.items():
                    if os.path.exists(file_path):
                        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)

                        # Select the best 30-second segment
                        audio = select_best_segment(audio, sr, target_length_samples)

                        # Save raw audio into HDF5
                        track_group.create_dataset(f'{stem}_audio', data=audio, dtype=np.float32)

                        # Compute and save Log-Mel spectrogram
                        log_mel_spectrogram = compute_log_mel_spectrogram(audio, sr, n_mels=n_mels, fmin=fmin, fmax=fmax)
                        track_group.create_dataset(f'{stem}_log_mel', data=log_mel_spectrogram, dtype=np.float32)
                    else:
                        print(f"File {file_path} not found.")

# Preprocess each split (train, test, valid)
print("Processing train split...")
preprocess_split(train_dir, train_hdf5_path)

print("Processing test split...")
preprocess_split(test_dir, test_hdf5_path)

print("Processing valid split...")
preprocess_split(valid_dir, valid_hdf5_path)

print("Preprocessing for all splits completed!")

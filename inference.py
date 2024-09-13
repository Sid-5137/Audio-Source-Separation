# inference.py
import torch
import torchaudio
from models.model import ModifiedUNet  # Import the model architecture
import os
import matplotlib.pyplot as plt
import numpy as np

# Define a function to load the trained model
def load_model(model_path, device):
    model = ModifiedUNet(in_channels=1, out_channels=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# Function to save spectrogram images
def save_spectrogram(mag, folder_path, filename):
    plt.figure(figsize=(10, 4))
    plt.imshow(mag.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {filename}')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'{filename}.png'))
    plt.close()
    print(f"Spectrogram saved: {filename}")

# Function to perform inference on a new track
def separate_sources(model, mixture_path, output_dir, n_fft=1024, hop_length=512):
    mixture, sr = torchaudio.load(mixture_path)
    if mixture.shape[0] > 1:
        mixture = torch.mean(mixture, dim=0, keepdim=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixture = mixture.to(device)
    mixture_spec = torch.stft(
        mixture, n_fft=n_fft, hop_length=hop_length, return_complex=True,
        window=torch.hann_window(n_fft).to(device)
    ).unsqueeze(0)
    mixture_mag = torch.abs(mixture_spec)

    # Save the true mixture spectrogram
    save_spectrogram(torch.log1p(mixture_mag[0]), output_dir, 'true_mixture')

    with torch.no_grad():
        output_mag = model(mixture_mag, target_shape=mixture_mag.shape)

    separated_audio = []
    stems = ['vocals', 'drums', 'bass', 'other']
    os.makedirs(output_dir, exist_ok=True)

    for i in range(4):
        output_spec = output_mag[:, i, :, :] * torch.exp(1j * torch.angle(mixture_spec.squeeze()))
        output_audio = torch.istft(output_spec, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(device))

        # Save predicted mask spectrogram
        save_spectrogram(torch.log1p(output_mag[0, i]), output_dir, f'predicted_mask_{stems[i]}')
        
        # Post-processing step: Example using median filtering (optional and adjustable)
        output_audio = torch.tensor(np.median(output_audio.cpu().numpy(), axis=-1)).to(device)
        output_audio = output_audio.squeeze() if output_audio.ndim > 1 else output_audio
        separated_audio.append(output_audio)

    for i, audio in enumerate(separated_audio):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        output_path = os.path.join(output_dir, f'{stems[i]}.wav')
        torchaudio.save(output_path, audio.cpu(), sample_rate=sr)
        print(f"Saved: {output_path}")

# Main function to run inference
def main():
    model_path = 'checkpoints/checkpoint_epoch_100.pth'
    mixture_path = r'D:\GPU_Projects\Prism2\Audio-Source-Separation_4stems\data\archive\test\AM Contra - Heart Peripheral\mixture.wav'
    output_dir = 'output_separated'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    separate_sources(model, mixture_path, output_dir)

if __name__ == "__main__":
    main()

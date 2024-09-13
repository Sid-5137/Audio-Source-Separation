# inference.py
import torch
import torchaudio
from models.model import ModifiedUNet  # Import the model architecture
import os

# Define a function to load the trained model
def load_model(model_path, device):
    # Initialize the model architecture with matching input and output channels used during training
    model = ModifiedUNet(in_channels=1, out_channels=4).to(device)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Check if the checkpoint contains additional state like optimizer or scaler
    if 'model_state_dict' in checkpoint:
        # Load the model state dictionary from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load directly if it's a simple state dict
        model.load_state_dict(checkpoint)

    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

# Function to perform inference on a new track
def separate_sources(model, mixture_path, output_dir, n_fft=1024, hop_length=512):
    # Load the mixture audio
    mixture, sr = torchaudio.load(mixture_path)
    device = 'cpu'
    window = torch.hann_window(n_fft, device=device)  # Define the window to reduce spectral leakage

    # If mixture has more than one channel, convert to mono (sum the channels)
    if mixture.shape[0] > 1:
        mixture = torch.mean(mixture, dim=0, keepdim=True)

    # Compute the spectrogram with a single channel
    mixture_spec = torch.stft(mixture, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window).unsqueeze(0).to(device)
    mixture_mag = torch.abs(mixture_spec)

    # Perform inference
    with torch.no_grad():
        output_mag = model(mixture_mag, target_shape=mixture_mag.shape)

    # Reconstruct phase and separate each component
    separated_audio = []
    for i in range(4):  # Four stems: vocals, drums, bass, other
        output_spec = output_mag[:, i, :, :] * torch.exp(1j * torch.angle(mixture_spec.squeeze()))
        output_audio = torch.istft(output_spec, n_fft=n_fft, hop_length=hop_length, window=window)

        # Ensure the output is 2D: [channels, samples]
        if output_audio.ndim == 1:
            output_audio = output_audio.unsqueeze(0)  # Add channel dimension if missing
        separated_audio.append(output_audio)

    # Save each separated audio component
    stems = ['vocals', 'drums', 'bass', 'other']
    os.makedirs(output_dir, exist_ok=True)
    for i, audio in enumerate(separated_audio):
        # Ensure the audio tensor has shape [channels, samples]
        audio = audio.squeeze() if audio.ndim == 3 else audio  # Remove unnecessary batch dimension if present
        output_path = os.path.join(output_dir, f'{stems[i]}.wav')
        torchaudio.save(output_path, audio, sample_rate=sr)
        print(f"Saved: {output_path}")

# Main function to run inference
def main():
    # Paths to model and new track
    model_path = 'checkpoints/checkpoint_epoch_100.pth'  # Update with the correct path to the saved model
    mixture_path = r'D:\GPU_Projects\Prism2\Audio-Source-Separation_4stems\data\archive\test\AM Contra - Heart Peripheral\mixture.wav'  # Update with the path to the new mixture track
    output_dir = 'output_separated'  # Directory where separated stems will be saved

    # Set device
    device = 'cpu'

    # Load the trained model
    model = load_model(model_path, device)

    # Run inference on the new track
    separate_sources(model, mixture_path, output_dir)

if __name__ == "__main__":
    main()

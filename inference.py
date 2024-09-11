# inference.py
import torch
import torchaudio
from models.model import UNet  # Import the model architecture
import os

# Define a function to load the trained model
def load_model(model_path, device):
    model = UNet(in_channels=1, out_channels=4).to(device)  # Ensure to match architecture used during training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

# Function to perform inference on a new track
def separate_sources(model, mixture_path, output_dir, n_fft=1024, hop_length=512):
    # Load the mixture audio
    mixture, sr = torchaudio.load(mixture_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixture_spec = torch.stft(mixture, n_fft=n_fft, hop_length=hop_length, return_complex=True).unsqueeze(0).to(device)
    mixture_mag = torch.abs(mixture_spec)

    # Perform inference
    with torch.no_grad():
        output_mag = model(mixture_mag)

    # Reconstruct phase and separate each component
    separated_audio = []
    for i in range(4):  # Four stems: vocals, drums, bass, other
        output_spec = output_mag[:, i, :, :] * torch.exp(1j * torch.angle(mixture_spec.squeeze()))
        output_audio = torch.istft(output_spec, n_fft=n_fft, hop_length=hop_length)
        separated_audio.append(output_audio)

    # Save each separated audio component
    stems = ['vocals', 'drums', 'bass', 'other']
    os.makedirs(output_dir, exist_ok=True)
    for i, audio in enumerate(separated_audio):
        output_path = os.path.join(output_dir, f'{stems[i]}.wav')
        torchaudio.save(output_path, audio.unsqueeze(0), sample_rate=sr)
        print(f"Saved: {output_path}")

# Main function to run inference
def main():
    # Paths to model and new track
    model_path = 'logs/model_epoch_10.pth'  # Update with the correct path to the saved model
    mixture_path = 'path/to/new/track/mixture.wav'  # Update with the path to the new mixture track
    output_dir = 'output_separated'  # Directory where separated stems will be saved

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model(model_path, device)

    # Run inference on the new track
    separate_sources(model, mixture_path, output_dir)

if __name__ == "__main__":
    main()


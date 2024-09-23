import torch
import librosa
import numpy as np
from model import UNet
import soundfile as sf
from utils import generate_log_mel_spectrogram, inverse_log_mel_spectrogram

def load_model(checkpoint_path, device='cuda'):
    # Load the trained model
    model = UNet(input_channels=1, output_channels=5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def process_audio_file(audio_file, model, device='cuda'):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=44100)
    
    # Generate log-mel spectrogram
    log_mel_spec = generate_log_mel_spectrogram(y, sr)
    
    # Convert to tensor and move to device
    log_mel_tensor = torch.tensor(log_mel_spec).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions

    # Perform inference
    with torch.no_grad():
        predicted_masks = model(log_mel_tensor)

    # Move prediction back to CPU and process the masks
    predicted_masks = predicted_masks.squeeze(0).cpu().numpy()

    # Convert masks back to audio (using inverse_log_mel_spectrogram)
    separated_sources = []
    for i in range(predicted_masks.shape[0]):
        source_audio = inverse_log_mel_spectrogram(predicted_masks[i], sr)
        separated_sources.append(source_audio)

    return separated_sources

def save_sources(sources, output_dir, sr=44100):
    # Save separated sources to files
    source_names = ['vocals', 'drums', 'bass', 'other', 'mixture']
    for i, source in enumerate(sources):
        output_path = f"{output_dir}/{source_names[i]}.wav"
        sf.write(output_path, source, sr)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    # Path to your audio file and trained model checkpoint
    audio_file = "path_to_your_audio_file.wav"
    checkpoint_path = "path_to_your_trained_model_checkpoint.pth"
    output_dir = "separated_sources"

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device=device)

    # Process the audio file and get separated sources
    separated_sources = process_audio_file(audio_file, model, device=device)

    # Save separated sources
    save_sources(separated_sources, output_dir)

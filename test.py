# test.py
import os
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import soundfile as sf
from models.unet import ModifiedUNet
from utils.audio_utils import chunk_spectrogram

def test_model_on_file(model, audio_path, device, output_dir="separated_audio", batch_size=32, device_ids=[0, 1]):
    model.eval()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)
    
    audio, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 44100:
        resampler = torchaudio.transforms.Resample(sample_rate, 44100)
        audio = resampler(audio)
    if audio.shape[0] != 2:
        raise ValueError("Input audio must be stereo (2 channels).")

    transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)
    complex_spec = transform(audio)
    magnitude_spec = complex_spec.abs()
    phase_spec = complex_spec.angle()
    mix_spec = torch.log1p(magnitude_spec)
    mix_chunks = chunk_spectrogram(mix_spec, 512)

    num_chunks = len(mix_chunks)
    batches = [mix_chunks[i:i + batch_size] for i in range(0, num_chunks, batch_size)]
    
    full_pred_specs = {s: [] for s in range(4)}
    with torch.no_grad():
        for batch in tqdm(batches, desc="Processing Chunks"):
            batch_tensor = torch.stack(batch).to(device)
            pred_spec = model(batch_tensor)
            pred_spec = pred_spec.cpu()
            for b in range(pred_spec.size(0)):
                for s in range(4):
                    full_pred_specs[s].append(torch.expm1(pred_spec[b, s]))

    target_time_frames = magnitude_spec.shape[-1]
    full_pred_magnitudes = [torch.cat(full_pred_specs[s], dim=-1)[..., :target_time_frames] for s in range(4)]
    full_mix_magnitude = torch.cat(mix_chunks, dim=-1)[..., :target_time_frames]
    full_mix_magnitude = torch.expm1(full_mix_magnitude)

    full_pred_specs = [mag * torch.exp(1j * phase_spec) for mag in full_pred_magnitudes]
    full_mix_spec = full_mix_magnitude * torch.exp(1j * phase_spec)

    inverse_transform = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)
    mix_audio = inverse_transform(full_mix_spec)
    source_audios = [inverse_transform(pred_spec) for pred_spec in full_pred_specs]

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    sf.write(f"{output_dir}/{file_name}_mix.wav", mix_audio.T.numpy(), 44100)
    for s, source_audio in enumerate(source_audios):
        source_name = ['vocals', 'drums', 'bass', 'other'][s]
        sf.write(f"{output_dir}/{file_name}_{source_name}.wav", source_audio.T.numpy(), 44100)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModifiedUNet(in_channels=2, out_channels=8)
    model.load_state_dict(torch.load("unet_model_epoch1.pth"))
    print("Loaded pretrained model from epoch 1")

    audio_path = "/home/sid/Desktop/Projects/Samsung_PRISM/Project_ausep/Code/musdb18hq/test/Al James - Schoolboy Facination/mixture.wav"
    test_model_on_file(model, audio_path, device, output_dir="separated_audio")
    print("Testing complete. Check 'separated_audio' folder for WAV files.")
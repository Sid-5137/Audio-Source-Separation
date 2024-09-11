# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import UNet
from dataset.dataset import MUSDB18HQDataset
import librosa
import numpy as np
from tqdm import tqdm
import mir_eval
import os
import matplotlib.pyplot as plt

# Function to calculate SDR using mir_eval
def calculate_sdr(estimates, references):
    sdr_values = []
    for i in range(len(estimates)):
        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
            references[i].cpu().numpy(), estimates[i].cpu().numpy()
        )
        sdr_values.append(sdr)
    return np.mean(sdr_values)

# Function to visualize and save spectrograms
def save_spectrogram(mag, folder_path, filename):
    try:
        plt.figure(figsize=(10, 4))
        plt.imshow(mag, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {filename}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, f'{filename}.png'))
        plt.close()
    except Exception as e:
        print(f"Error saving spectrogram {filename}: {e}")

# Custom collate function to handle variable-length spectrograms
def custom_collate_fn(batch):
    inputs_mag = [item[0] for item in batch]
    targets_mag = [item[1] for item in batch]
    max_len = max([input.shape[1] for input in inputs_mag])

    padded_inputs = [torch.nn.functional.pad(input, (0, max_len - input.shape[1])) for input in inputs_mag]
    padded_targets = [torch.nn.functional.pad(target, (0, max_len - target.shape[2])) for target in targets_mag]

    inputs_mag = torch.stack(padded_inputs)
    targets_mag = torch.stack(padded_targets)

    other_elements = [[item[i] for item in batch] for i in range(2, len(batch[0]))]

    return (inputs_mag, targets_mag, *other_elements)

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_losses = []
    sdr_values = []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation") as pbar:
            for inputs_mag, targets_mag, inputs_spec, vocals_spec, drums_spec, bass_spec, other_spec in pbar:
                inputs_mag = inputs_mag.unsqueeze(1).to(device)
                targets_mag = targets_mag.to(device)

                outputs_mag = model(inputs_mag)
                loss = criterion(outputs_mag, targets_mag)
                val_losses.append(loss.item())

                estimates = torch.cat([outputs_mag[:, i, :, :].unsqueeze(0) for i in range(4)], dim=0)
                references = torch.cat([targets_mag[:, i, :, :].unsqueeze(0) for i in range(4)], dim=0)
                sdr = calculate_sdr(estimates, references)
                sdr_values.append(sdr)

                pbar.set_postfix(loss=loss.item(), sdr=sdr)

    avg_val_loss = np.mean(val_losses)
    avg_sdr = np.mean(sdr_values)

    return avg_val_loss, avg_sdr

# Load Data
root_dir = 'data/archive/train'
val_dir = 'data/archive/valid'

if not os.path.exists(root_dir):
    raise FileNotFoundError(f"Training directory not found: {root_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

train_dataset = MUSDB18HQDataset(root_dir)
val_dataset = MUSDB18HQDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

log_dir = 'logs/'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_log.txt')
sdr_file = os.path.join(log_dir, 'sdr_values.txt')

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_losses = []

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]") as pbar:
        for i, (inputs_mag, targets_mag, inputs_spec, vocals_spec, drums_spec, bass_spec, other_spec) in enumerate(pbar):
            inputs_mag = inputs_mag.unsqueeze(1).to(device)
            targets_mag = targets_mag.to(device)

            outputs_mag = model(inputs_mag)
            loss = criterion(outputs_mag, targets_mag)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

            if i % 10 == 0:
                track_folder = os.path.join(log_dir, f'track_{epoch+1}_{i}')
                os.makedirs(track_folder, exist_ok=True)

                input_spectrogram = inputs_mag[0].squeeze(0).cpu().numpy()
                save_spectrogram(np.log1p(input_spectrogram), track_folder, 'input_mixture')

                stems = ['vocals', 'drums', 'bass', 'other']
                for j in range(4):
                    mask = outputs_mag[0, j].detach().cpu().numpy()
                    save_spectrogram(np.log1p(mask), track_folder, f'predicted_mask_{stems[j]}')

                    true_spectrogram = targets_mag[0, j].detach().cpu().numpy()
                    save_spectrogram(np.log1p(true_spectrogram), track_folder, f'true_{stems[j]}')

    avg_train_loss = np.mean(train_losses)

    # Validation
    avg_val_loss, avg_sdr = validate(model, val_loader, criterion, device)

    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}\n")
        f.write(f"Training Loss: {avg_train_loss:.4f}\n")
        f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
        f.write(f"Average SDR: {avg_sdr:.4f}\n\n")

    with open(sdr_file, 'a') as f:
        f.write(f"Epoch {epoch+1}: {avg_sdr:.4f}\n")

    model_save_path = os.path.join(log_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

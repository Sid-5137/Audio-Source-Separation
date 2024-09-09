# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import UNet
from dataset.dataset import MUSDB18HQDataset
import torchaudio
import numpy as np
from tqdm import tqdm
import mir_eval  # Make sure to install this library using `pip install mir_eval`
import os

# Function to calculate SDR using mir_eval
def calculate_sdr(estimates, references):
    sdr_values = []
    for i in range(len(estimates)):
        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
            references[i].cpu().numpy(), estimates[i].cpu().numpy()
        )
        sdr_values.append(sdr)
    return np.mean(sdr_values)

# Load Data
root_dir = 'path/to/musdb18hq/train'  # Update with your MUSDB18-HQ dataset path
val_dir = 'path/to/musdb18hq/val'    # Add a validation set path if available
train_dataset = MUSDB18HQDataset(root_dir)
val_dataset = MUSDB18HQDataset(val_dir)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize log files
log_dir = 'logs/'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_log.txt')
sdr_file = os.path.join(log_dir, 'sdr_values.txt')

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]") as pbar:
        for inputs_mag, targets_mag, inputs_spec, vocals_spec, drums_spec, bass_spec, other_spec in pbar:
            inputs_mag, targets_mag = inputs_mag.to(device), targets_mag.to(device)

            # Forward pass
            outputs_mag = model(inputs_mag)
            loss = criterion(outputs_mag, targets_mag)
            train_losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

    avg_train_loss = np.mean(train_losses)

    # Validation phase
    model.eval()
    val_losses = []
    sdr_values = []
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]") as pbar:
            for inputs_mag, targets_mag, inputs_spec, vocals_spec, drums_spec, bass_spec, other_spec in pbar:
                inputs_mag, targets_mag = inputs_mag.to(device), targets_mag.to(device)

                # Forward pass
                outputs_mag = model(inputs_mag)
                loss = criterion(outputs_mag, targets_mag)
                val_losses.append(loss.item())

                # Calculate SDR values for validation
                estimates = torch.cat([outputs_mag[:, i, :, :].unsqueeze(0) for i in range(4)], dim=0)
                references = torch.cat([targets_mag[:, i, :, :].unsqueeze(0) for i in range(4)], dim=0)
                sdr = calculate_sdr(estimates, references)
                sdr_values.append(sdr)

                pbar.set_postfix(loss=loss.item(), sdr=sdr)

    avg_val_loss = np.mean(val_losses)
    avg_sdr = np.mean(sdr_values)

    # Log results
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}\n")
        f.write(f"Training Loss: {avg_train_loss:.4f}\n")
        f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
        f.write(f"Average SDR: {avg_sdr:.4f}\n\n")

    with open(sdr_file, 'a') as f:
        f.write(f"Epoch {epoch+1}: {avg_sdr:.4f}\n")

    # Save the trained model after each epoch
    model_save_path = os.path.join(log_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")


# Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from models.model import ModifiedUNet
from dataset.dataset import MUSDB18HQDataset
import numpy as np
from tqdm import tqdm
import mir_eval
import os
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
import torch.multiprocessing as mp

# Define a small epsilon to avoid log of zero or negative numbers
EPSILON = 1e-8

# Initialize DDP
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'Seedlab-Server1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Cleanup DDP
def cleanup():
    dist.destroy_process_group()

# Function to ensure 2D shape for SDR calculation
def ensure_2d(array):
    """Ensure the array is 2D by adding a new axis if necessary."""
    if array.ndim == 1:
        return array[np.newaxis, :]
    return array

# Function to pad or truncate sequences to a consistent length
def pad_or_truncate(array, length):
    """Pad with zeros or truncate the array to the specified length."""
    if array.shape[-1] < length:
        return np.pad(array, ((0, 0), (0, length - array.shape[-1])), mode='constant')
    elif array.shape[-1] > length:
        return array[:, :length]
    return array

# Function to calculate SDR using mir_eval
def calculate_sdr(estimates, references):
    sdr_values = []
    for i in range(len(estimates)):
        # Convert PyTorch tensors to NumPy arrays, ensuring they are 2D
        est = ensure_2d(estimates[i].cpu().numpy()) if isinstance(estimates[i], torch.Tensor) else ensure_2d(estimates[i])
        ref = ensure_2d(references[i].numpy()) if isinstance(references[i], torch.Tensor) else ensure_2d(references[i])

        # Compute SDR
        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(ref, est)
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

# Validation function with transformations to ensure consistent dimensions
def validate(model, val_loader, criterion, device):
    model.eval()
    val_losses = []
    sdr_values = []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation") as pbar:
            for inputs_mag, targets_mag, inputs_spec, vocals_spec, drums_spec, bass_spec, other_spec, mixture_name in pbar:
                inputs_mag = inputs_mag.unsqueeze(1).to(device)
                targets_mag = targets_mag.to(device)

                # Resize outputs to ensure the shapes match
                outputs_mag = model(inputs_mag, target_shape=targets_mag.shape)
                if outputs_mag.shape != targets_mag.shape:
                    outputs_mag = torch.nn.functional.interpolate(outputs_mag, size=targets_mag.shape[-2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs_mag, targets_mag)
                val_losses.append(loss.item())

                estimates = torch.cat([outputs_mag[:, i, :, :].unsqueeze(0) for i in range(4)], dim=0)
                references = torch.cat([targets_mag[:, i, :, :].unsqueeze(0) for i in range(4)], dim=0)
                
                # Transform references and estimates for consistent dimensions
                estimates = [ensure_2d(e.cpu().numpy()) for e in estimates]
                references = [ensure_2d(r.cpu().numpy()) for r in references]

                # Adjust lengths to match
                max_length = min(min([e.shape[-1] for e in estimates]), min([r.shape[-1] for r in references]))
                estimates = [pad_or_truncate(e, max_length) for e in estimates]
                references = [pad_or_truncate(r, max_length) for r in references]

                sdr = calculate_sdr(estimates, references)
                sdr_values.append(sdr)

                pbar.set_postfix(loss=loss.item(), sdr=sdr)

    avg_val_loss = np.mean(val_losses)
    avg_sdr = np.mean(sdr_values)

    return avg_val_loss, avg_sdr

import os

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, scaler, rank):
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_rank_{rank}.pth')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, scaler, rank):
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # List all checkpoint files in the 'checkpoints' directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch') and f.endswith(f'_rank_{rank}.pth')]
    
    if not checkpoint_files:
        return 1  # Start from epoch 1 if no checkpoints are found
    
    # Find the latest checkpoint based on creation time
    latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoint_files], key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    print(f"Resumed training from {latest_checkpoint}")
    
    return checkpoint['epoch'] + 1 

# Main training loop wrapped in the function
def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Load Data
    root_dir = 'data/archive/train'
    val_dir = 'data/archive/valid'

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Training directory not found: {root_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_dataset = MUSDB18HQDataset(root_dir)
    val_dataset = MUSDB18HQDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)

    device = torch.device(f"cuda:{rank}")
    model = ModifiedUNet().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # Added find_unused_parameters=True

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # Load from checkpoint if available
    start_epoch = load_checkpoint(model, optimizer, scaler, rank)

    num_epochs = 10
    accumulation_steps = 4

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_losses = []

        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training] Rank {rank}") as pbar:
            for i, (inputs_mag, targets_mag, inputs_spec, vocals_spec, drums_spec, bass_spec, other_spec, mixture_name) in enumerate(pbar):
                inputs_mag = inputs_mag.unsqueeze(1).to(device)
                targets_mag = targets_mag.to(device)

                with autocast():
                    outputs_mag = model(x=inputs_mag, target_shape=targets_mag.shape)
                    loss = criterion(outputs_mag, targets_mag) / accumulation_steps
                    train_losses.append(loss.item())

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()

                pbar.set_postfix(loss=loss.item())

                track_name = mixture_name[0].split('.')[0]
                track_folder = os.path.join('logs',f'{track_name}_epoch_{epoch}_rank_{rank}')
                os.makedirs(track_folder, exist_ok=True)

                input_spectrogram = inputs_mag[0].squeeze(0).cpu().numpy()
                save_spectrogram(np.log1p(np.maximum(input_spectrogram, EPSILON)), track_folder, 'input_mixture')

                stems = ['vocals', 'drums', 'bass', 'other']
                for j in range(4):
                    mask = outputs_mag[0, j].detach().cpu().numpy()
                    save_spectrogram(np.log1p(np.maximum(mask, EPSILON)), track_folder, f'predicted_mask_{stems[j]}')

                    true_spectrogram = targets_mag[0, j].detach().cpu().numpy()
                    save_spectrogram(np.log1p(np.maximum(true_spectrogram, EPSILON)), track_folder, f'true_{stems[j]}')

        avg_train_loss = np.mean(train_losses)
        avg_val_loss, avg_sdr = validate(model, val_loader, criterion, device)

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, scaler, rank)

        # Log results
        with open(f'training_log_rank_{rank}.txt', 'a') as f:
            f.write(f"Epoch {epoch}/{num_epochs}\n")
            f.write(f"Training Loss: {avg_train_loss:.4f}\n")
            f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
            f.write(f"Average SDR: {avg_sdr:.4f}\n\n")

        with open(f'sdr_values_rank_{rank}.txt', 'a') as f:
            f.write(f"Epoch {epoch}: {avg_sdr:.4f}\n")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

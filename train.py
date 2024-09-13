# Imports (same as your script)
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
from torch.utils.checkpoint import checkpoint

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

def ensure_2d(array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    if array.ndim == 1:
        return array.reshape(1, -1)
    elif array.ndim == 2:
        return array
    else:
        return np.reshape(array, (array.shape[0], -1))

def calculate_sdr(estimates, references):
    sdr_values = []
    for i in range(len(estimates)):
        ref = ensure_2d(references[i])
        est = ensure_2d(estimates[i])
        if np.all(ref == 0):
            ref += 1e-8
        if np.all(est == 0):
            est += 1e-8
        min_length = min(ref.shape[1], est.shape[1])
        ref = ref[:, :min_length]
        est = est[:, :min_length]
        try:
            sdr, _, _, _ = mir_eval.separation.bss_eval_sources(ref, est)
            sdr_values.append(sdr)
        except ValueError as e:
            print(f"Error calculating SDR for index {i}: {e}")
            sdr_values.append(-np.inf)
    valid_sdr_values = [s for s in sdr_values if s != -np.inf]
    return np.mean(valid_sdr_values) if valid_sdr_values else -np.inf

def save_spectrogram(mag, folder_path, filename):
    try:
        plt.figure(figsize=(10, 4))
        plt.imshow(mag, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {filename}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()
        save_path = os.path.join(folder_path, f'{filename}.png')
        plt.savefig(save_path)
        print(f"Spectrogram saved: {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error saving spectrogram {filename}: {e}")

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

def validate(model, val_loader, criterion, device):
    model.eval()
    val_losses = []
    sdr_values = []
    with torch.no_grad():
        with tqdm(val_loader, desc="Validation") as pbar:
            for inputs_mag, targets_mag, inputs_spec, vocals_spec, drums_spec, bass_spec, other_spec, mixture_name in pbar:
                inputs_mag = inputs_mag.unsqueeze(1).to(device)
                targets_mag = targets_mag.to(device)
                outputs_mag = model(inputs_mag, target_shape=targets_mag.shape)
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

def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    root_dir = 'data/archive/train'
    val_dir = 'data/archive/valid'

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Training directory not found: {root_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_dataset = MUSDB18HQDataset(root_dir)
    val_dataset = MUSDB18HQDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)

    device = torch.device(f"cuda:{rank}")
    model = ModifiedUNet().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    criterion = torch.nn.MSELoss()  # Replace or augment this with other loss functions as discussed
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('checkpoint_epoch') and f.endswith(f'_rank_{rank}.pth')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2]))

    start_epoch = 0
    if checkpoint_files:
        latest_checkpoint = os.path.join('checkpoints', checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from {latest_checkpoint}")

    num_epochs = 1000
    accumulation_steps = 4

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = []

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training] Rank {rank}") as pbar:
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

                # Save multiple spectrograms
                if i < 5 and rank == 0:  # Save spectrograms for the first 5 batches of each epoch
                    track_name = mixture_name[0].split('.')[0]
                    track_folder = os.path.join('logs', f'{track_name}_epoch_{epoch+1}_rank_{rank}')
                    os.makedirs(track_folder, exist_ok=True)

                    input_spectrogram = inputs_mag[0].squeeze(0).detach().cpu().numpy()
                    save_spectrogram(np.log1p(np.maximum(input_spectrogram, EPSILON)), track_folder, f'input_mixture_{i}')

                    stems = ['vocals', 'drums', 'bass', 'other']
                    for j in range(4):
                        mask = outputs_mag[0, j].detach().cpu().numpy()
                        save_spectrogram(np.log1p(np.maximum(mask, EPSILON)), track_folder, f'predicted_mask_{stems[j]}_{i}')

                        true_spectrogram = targets_mag[0, j].detach().cpu().numpy()
                        save_spectrogram(np.log1p(np.maximum(true_spectrogram, EPSILON)), track_folder, f'true_{stems[j]}_{i}')

        avg_train_loss = np.mean(train_losses)
        avg_val_loss, avg_sdr = validate(model, val_loader, criterion, device)

        # Save model checkpoint only from rank 0
        if rank == 0:
            model_save_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, model_save_path)
            print(f"Model saved at {model_save_path}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

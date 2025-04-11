import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import PrecomputedMusdbDataset, preprocess_dataset
from models.unet import ModifiedUNet
from utils.audio_utils import StereoSpectrogramLoss


def train(model, dataloader, criterion, optimizer, epochs=10, device_ids=[0, 1]):
    model.train()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        running_loss = 0.0
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, unit="batch", total=len(dataloader))
        for i, (mix_batch, target_batch) in enumerate(batch_iterator):
            try:
                mix_batch, target_batch = mix_batch.to(device), target_batch.to(device)
                optimizer.zero_grad()
                pred_spec = model(mix_batch)
                loss = criterion(pred_spec, target_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            except Exception as e:
                tqdm.write(f"Crash at Batch {i}: {str(e)}")
                raise

        avg_loss = running_loss / len(dataloader)
        tqdm.write(f"Epoch {epoch + 1}/{epochs} Completed - Average Loss: {avg_loss:.4f}")
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
                   f"unet_model_epoch{epoch + 1}.pth")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess_dataset("./musdb18hq", "musdb_specs")
    dataset = PrecomputedMusdbDataset(spec_dir="musdb_specs")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    model = ModifiedUNet(in_channels=2, out_channels=8)
    criterion = StereoSpectrogramLoss(w_log_mag=1.0, w_lin_mag=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print(f"Total chunks in training dataset: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    train(model, dataloader, criterion, optimizer, epochs=10)
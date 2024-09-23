import torch
from torch.utils.data import DataLoader
from model import UNet
from dataset import MUSDBDataset
from utils import sdr_loss
from tqdm import tqdm

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        batch_bar = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
        for batch_idx, (mixture, stems) in enumerate(batch_bar):
            mixture, stems = mixture.to(device), stems.to(device)
            predicted_masks = model(mixture)

            # Loss based on SDR (or any other metrics)
            loss = sdr_loss(predicted_masks, stems)
            total_loss += loss.item()
            batch_bar.set_postfix({"Batch Loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"Average SDR Loss: {avg_loss}")
    return avg_loss

if __name__ == "__main__":
    # Load the trained model
    model = UNet(input_channels=1, output_channels=5)
    model.load_state_dict(torch.load('unet_music_source_separation_final.pth'))
    model = model.to('cuda')

    # Load the evaluation dataset
    val_dataset = MUSDBDataset('data/musdb18_test.h5')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate the model
    evaluate_model(model, val_loader)

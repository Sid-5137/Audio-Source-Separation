# test.py
import torch
import torchaudio
from torch.utils.data import DataLoader
from models.model import UNet
from dataset.dataset import MUSDB18HQDataset
import numpy as np
from tqdm import tqdm
import mir_eval
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

# Load the trained model
def load_model(model_path, device):
    model = UNet(in_channels=1, out_channels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

# Function to test the model on the test dataset
def test_model(model, test_loader, device):
    model.eval()
    sdr_values = []

    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as pbar:
            for inputs_mag, targets_mag, inputs_spec, vocals_spec, drums_spec, bass_spec, other_spec in pbar:
                inputs_mag, targets_mag = inputs_mag.to(device), targets_mag.to(device)

                # Forward pass
                outputs_mag = model(inputs_mag)

                # Calculate SDR values for testing
                estimates = torch.cat([outputs_mag[:, i, :, :].unsqueeze(0) for i in range(4)], dim=0)
                references = torch.cat([targets_mag[:, i, :, :].unsqueeze(0) for i in range(4)], dim=0)
                sdr = calculate_sdr(estimates, references)
                sdr_values.append(sdr)

                pbar.set_postfix(sdr=sdr)

    avg_sdr = np.mean(sdr_values)
    print(f"Average SDR on Test Set: {avg_sdr:.4f}")
    return avg_sdr

# Main function to run testing
def main():
    # Paths to model and test dataset
    model_path = 'logs/model_epoch_10.pth'  # Update with the correct path to the saved model
    test_dir = 'path/to/musdb18hq/test'    # Update with the correct path to the test dataset

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset = MUSDB18HQDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Load the trained model
    model = load_model(model_path, device)

    # Test the model and log results
    avg_sdr = test_model(model, test_loader, device)

    # Save test results
    log_dir = 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    test_log_file = os.path.join(log_dir, 'test_results.txt')
    with open(test_log_file, 'a') as f:
        f.write(f"Average SDR on Test Set: {avg_sdr:.4f}\n")

if __name__ == "__main__":
    main()


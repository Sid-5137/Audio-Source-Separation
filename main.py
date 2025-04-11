# main.py
import argparse
from train import train
from test import test_model_on_file
from data.dataset import PrecomputedMusdbDataset, preprocess_dataset
from models.unet import ModifiedUNet
from utils.audio_utils import StereoSpectrogramLoss
from torch.utils.data import DataLoader
import torch

def main():
    parser = argparse.ArgumentParser(description="Audio Separation with U-Net")
    parser.add_argument('--mode', choices=['preprocess', 'train', 'test'], required=True, help="Mode to run")
    parser.add_argument('--musdb_root', type=str, default="./musdb18hq", help="Path to musdb18hq")
    parser.add_argument('--spec_dir', type=str, default="musdb_specs", help="Directory for precomputed spectrograms")
    parser.add_argument('--audio_path', type=str, help="Path to test audio file (for test mode)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of tracks to preprocess")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModifiedUNet(in_channels=2, out_channels=8)

    if args.mode == "preprocess":
        num_tracks = preprocess_dataset(args.musdb_root, args.spec_dir, limit=args.limit)
        print(f"Preprocessing completed. Processed {num_tracks} tracks.")
    elif args.mode == "train":
        dataset = PrecomputedMusdbDataset(spec_dir=args.spec_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        criterion = StereoSpectrogramLoss(w_log_mag=1.0, w_lin_mag=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        print(f"Total chunks: {len(dataset)}, Batches per epoch: {len(dataloader)}")
        train(model, dataloader, criterion, optimizer, epochs=args.epochs)
    elif args.mode == "test":
        if not args.audio_path:
            raise ValueError("Audio path required for testing.")
        model.load_state_dict(torch.load("unet_model_epoch1.pth"))
        test_model_on_file(model, args.audio_path, device, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
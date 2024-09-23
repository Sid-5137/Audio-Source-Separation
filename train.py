import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MUSDBDataset
from model import UNet
from utils import sdr_loss
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
from torchsummary import summary

def save_checkpoint(model, optimizer, epoch, path):
    """Helper function to save model checkpoints."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at epoch {epoch}.")

def train(model, dataloader, val_loader, optimizer, scheduler, num_epochs=20, device='cuda', save_interval=5, output_dir='./', accumulation_steps=1):
    model.train()
    scaler = GradScaler()  # Initialize GradScaler for AMP
    log_file = os.path.join(output_dir, 'training_log.txt')  # Log file path

    # Open the log file in append mode
    with open(log_file, 'a') as log:
        log.write("Epoch, Training Loss, Validation Loss\n")  # Add headers for the log

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
            optimizer.zero_grad()

            for batch_idx, (mixture, stems) in enumerate(batch_bar):
                mixture, stems = mixture.to(device), stems.to(device)

                # Forward pass with AMP
                with autocast(device_type='cuda', dtype=torch.float16):  # Enable AMP for forward pass
                    predicted_masks = model(mixture)
                    loss = sdr_loss(predicted_masks, stems) / accumulation_steps  # Scale the loss

                # Backward pass with AMP and gradient accumulation
                scaler.scale(loss).backward()

                # Update the model after 'accumulation_steps' mini-batches
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # Reset gradients after updating

                # Track loss
                epoch_loss += loss.item() * accumulation_steps  # Multiply back to undo scaling
                batch_bar.set_postfix({"Batch Loss": loss.item() * accumulation_steps})

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss}")

            # Validation after each epoch
            validation_loss = evaluate(model, val_loader, device)

            # Log the epoch details
            log.write(f"{epoch+1}, {avg_epoch_loss}, {validation_loss}\n")

            # Step the learning rate scheduler based on validation loss
            scheduler.step(validation_loss)

            # Save checkpoint every 'save_interval' epochs
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch+1}.pth"
                save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete.")

def evaluate(model, dataloader, device='cuda'):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        batch_bar = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
        for batch_idx, (mixture, stems) in enumerate(batch_bar):
            mixture, stems = mixture.to(device), stems.to(device)
            predicted_masks = model(mixture)
            loss = sdr_loss(predicted_masks, stems)
            total_loss += loss.item()
            batch_bar.set_postfix({"Batch Loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss

def run_training(train_data_path, val_data_path, batch_size, epochs, output_dir, accumulation_steps=1, save_interval=5):
    # Load dataset with all tracks
    train_dataset = MUSDBDataset(train_data_path)
    val_dataset = MUSDBDataset(val_data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load model
    model = UNet(input_channels=1, output_channels=5)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    
    model = model.to('cuda')  # Move model to GPU

    # Print the model summary
    summary(model, input_size=(1, 128, 2584))  # Assuming input size based on your data

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler (optional but recommended)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Train the model with progress tracking
    train(model, train_loader, val_loader, optimizer, scheduler, num_epochs=epochs, output_dir=output_dir, accumulation_steps=accumulation_steps, save_interval=save_interval)
    
    # Save final model
    torch.save(model.state_dict(), f'{output_dir}/unet_music_source_separation_final.pth')
    print(f"Final model saved to {output_dir}/unet_music_source_separation_final.pth")

import optuna
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MUSDBDataset
from model import UNet
from utils import sdr_loss
from train import train, evaluate

def objective(trial, train_data_path, val_data_path):
    # Define hyperparameters for tuning
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])

    # Load dataset
    train_dataset = MUSDBDataset(train_data_path)
    val_dataset = MUSDBDataset(val_data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = UNet(input_channels=1, output_channels=5).to('cuda')

    # Optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
    # Train model for a few epochs to evaluate
    train(model, train_loader, optimizer, None, num_epochs=5, device='cuda')

    # Evaluate on the validation set
    validation_loss = evaluate(model, val_loader, device='cuda')
    
    # Return the validation loss to Optuna for optimization
    return validation_loss

def run_optuna_tuning(train_data_path, val_data_path):
    # Create the study object and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data_path, val_data_path), n_trials=50)

    # Output the best hyperparameters
    print("Best Hyperparameters: ", study.best_params)

import argparse
from train import run_training  # Import the run_training function

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='U-Net for Music Source Separation')

    # Add arguments for batch size, epochs, dataset directory, and model output directory
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
    parser.add_argument('--train_data', type=str, default='data/musdb18_train_30sec.h5', help='Path to the training dataset')
    parser.add_argument('--val_data', type=str, default='data/musdb18_valid_30sec.h5', help='Path to the validation dataset')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save model checkpoints and final model')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of steps for gradient accumulation')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning with Optuna')
    parser.add_argument('--save_interval', type=int, default=5, help='Interval (in epochs) to save model checkpoints')


    # Parse the arguments
    args = parser.parse_args()

    if args.tune:
        # Run hyperparameter tuning with passed arguments
        from optuna_tuning import run_optuna_tuning
        run_optuna_tuning(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
        )
    else:
        # Regular training
        run_training(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
            accumulation_steps=args.accumulation_steps,
            save_interval=args.save_interval
        )

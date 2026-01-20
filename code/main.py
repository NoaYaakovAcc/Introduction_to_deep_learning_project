import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm # Import from your code

# --- Import custom modules ---
from data import scan_game, ChessBoardDataset
import plot # Import from your code
from eval_utils import evaluate_full_board_accuracy
from model import ChessNet
from train_utils import train_one_epoch, validate


def get_all_files_in_dirs(data_root, directory_list):
    all_samples = []
    for dir in directory_list:
        path = os.path.join(data_root, dir)
        # Assuming CSV filename matches game folder name prefix (e.g., game2.csv)
        csv_path = os.path.join(path, f"{dir.split('_')[0]}.csv")
        print(f"Looking for data in: {path} | CSV: {csv_path}")
        all_samples.extend(scan_game(path, csv_path))

    return all_samples


def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ==========================================
# YOUR MAIN FUNCTION (From Your Code)
# ==========================================
def main():
    # 1.1 Parameters chosen for this run
    RESOLUTION = 480  # Global parameter for image resolution (X*X)
    train_games_numbers = [2,4,5,6]
    val_games_numbers = [7]
    out = 'experiments'
    epochs = 50
    batch = 32
    lr = 0.001
    mode_type = 1  # 0 for zero_shot, 1 for finetune
    real_percent = 0.1  # Used only in finetune mode
    have_args = True

    #1.2 Parse command line arguments if needed
    if have_args:
        parser = argparse.ArgumentParser(description="Train ChessNet with STN")
        parser.add_argument('--data_root', type=str, required=True, help='Path to data folder')
        parser.add_argument('--val_games', nargs='+', required=True, help='List of games for validation folders')
        parser.add_argument('--train_games', nargs='+', required=True, help='List of games for train folders')
        parser.add_argument('--out', type=str, default='experiments', help='Output folder')
        parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
        parser.add_argument('--batch', type=int, default=32, help='Batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--mode', type=str, choices=['zero_shot', 'finetune'], default='zero_shot', help='Training mode')
        parser.add_argument('--real_percent', type=float, default=0.1, help='Fraction of real data to use in train for finetune')
        
        args = parser.parse_args()

    #1.3 parameter assignment
        data_root = args.data_root
        train_games_numbers = args.train_games
        val_games_numbers = args.val_games
        out = args.out
        epochs = args.epochs
        batch = args.batch
        lr = args.lr
        mode = args.mode
        real_percent = args.real_percent
    else:
    #1.4 parameter adjustments
        mode = 'finetune' if mode_type == 1 else 'zero_shot'
        data_root = r'data'
        train_games = []
        for game in train_games_numbers:
            train_games.append(f'game{game}_per_frame')
        val_games = []
        for game in val_games_numbers:
            val_games.append(f'game{game}_per_frame')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} with mode: {mode}")
    set_seed()

    # 2. Data Preparation
    all_train_samples = get_all_files_in_dirs(data_root, train_games)
    all_val_samples = get_all_files_in_dirs(data_root, val_games)
    # Split by domain
    synthetic_train_samples = [s for s in all_train_samples if s.domain == 'synthetic']
    real_train_samples = [s for s in all_train_samples if s.domain == 'real']
    synthetic_val_samples = [s for s in all_val_samples if s.domain == 'synthetic']
    real_val_samples = [s for s in all_val_samples if s.domain == 'real']

    
    # Logic for Training Mode
    if mode == 'zero_shot':
        # Train on Synthetic ONLY, Validate on Real
        train_samples = synthetic_train_samples
        val_samples = real_val_samples
        folder_name = "visual_results_zero_shot"
        
    elif mode == 'finetune':
        # Train on Synthetic + Small % of Real, Validate on remaining Real
        random.shuffle(real_train_samples)
        n_real_train = int((real_percent * len(synthetic_train_samples))/(1-real_percent))
        real_train = real_train_samples[:n_real_train]
        real_val = real_val_samples
        
        train_samples = synthetic_train_samples + real_train
        val_samples = real_val
        folder_name = f"visual_results_finetune_{int(real_percent*100)}"
    
    print(f"Training set size: {len(train_samples)}")
    print(f"Validation set size: {len(val_samples)}")
    
    # 3. Transforms and Loaders
    transform = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
    ])
    
    train_ds = ChessBoardDataset(train_samples, transform=transform)
    val_ds = ChessBoardDataset(val_samples, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)
    
    # 4. Model Initialization
    # Using the custom ChessNet from model.py
    model = ChessNet(num_classes=13, resolution=RESOLUTION).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        # Using functions from train_utils.py
        model, train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # 6. Final Evaluation
    print("Training Complete. Evaluating Full Board Accuracy...")
    plot.plot_list(train_acc, "Loss", "Epochs", 
                   f"Training miss rate over Epochs with real data precentage {real_percent*100}%", 
                   save_dir=folder_name)
    plot.plot_list(val_acc, "Loss", "Epochs", 
                   f"Validation miss rate over Epochs with real data precentage {real_percent*100}%", 
                   save_dir=folder_name)
    evaluate_full_board_accuracy(model, val_loader, device, folder_name=folder_name)
    
    # 7. Save Model Weights (CRITICAL step for predict_board to work)
    MODEL_PATH = 'best_model.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model weights saved successfully to {MODEL_PATH}")

if __name__ == '__main__':
    main()
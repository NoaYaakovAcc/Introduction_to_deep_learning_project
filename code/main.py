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
import data


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


def get_games(game_numbers):
    games = []
    for game in game_numbers:
        games.append(f'game{game}_per_frame')
    return games

# ==========================================
# YOUR MAIN FUNCTION (From Your Code)
# ==========================================
def main():
    # 1.1 Parameters chosen for this run
    RESOLUTION = 480  # Global parameter for image resolution (X*X)
    synthetic_train_samples_train_games_numbers = [1]
    real_train_samples_train_games_numbers = [2,4,5]
    val_games_numbers = [6,7]

    out = 'experiments'
    synthetic_epochs = 20
    real_epochs = 5
    batch = 16
    lr = 0.001
    have_args = False
    add_blur = False
    add_noise = False
    #1.2 Parse command line arguments if needed
    if have_args:
        parser = argparse.ArgumentParser(description="Train ChessNet with STN")
        parser.add_argument('--data_root', type=str, required=True, help='Path to data folder')
        parser.add_argument('--val_games', nargs='+', required=True, help='List of games for validation folders')
        parser.add_argument('--syntetic_train_games', nargs='+', required=True, help='List of games for train folders')
        parser.add_argument('--real_train_games', nargs='+', required=True, help='List of games for train folders')
        parser.add_argument('--out', type=str, default='experiments', help='Output folder')
        parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
        parser.add_argument('--batch', type=int, default=32, help='Batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--real_epochs', type=int, default=1, help='number of epochs to train on real data')
        
        args = parser.parse_args()

    #1.3 parameter assignment
        data_root = args.data_root
        synthetic_train_samples_train_games_numbers = args.syntetic_train_games
        real_train_samples_train_games_numbers = args.real_train_games
        val_games = args.val_games
        out = args.out
        synthetic_epochs = args.epochs
        batch = args.batch
        lr = args.lr
        mode = args.mode
        real_epochs = args.real_epochs
    else:
    #1.4 parameter adjustments
        data_root = r'data'
        synthetic_train_games = get_games(synthetic_train_samples_train_games_numbers)
        real_train_games = get_games(real_train_samples_train_games_numbers)
        val_games = get_games(val_games_numbers)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    set_seed()

    # 2. Data Preparation
    synthetic_train_samples = get_all_files_in_dirs(data_root, synthetic_train_games)
    real_train_samples = get_all_files_in_dirs(data_root, real_train_games)
    all_val_samples = get_all_files_in_dirs(data_root, val_games)
    # Split by domain
    synthetic_train_samples = [s for s in synthetic_train_samples if s.domain == 'synthetic']
    real_train_samples = [s for s in real_train_samples if s.domain == 'real']
    synthetic_val_samples = [s for s in all_val_samples if s.domain == 'synthetic']
    real_val_samples = [s for s in all_val_samples if s.domain == 'real']

    # Logic for Training Mode
    if real_epochs != 0:
        folder_name = "visual_results_zero_shot"
        
    elif mode == 'finetune':
        # Train on Synthetic + Small % of Real, Validate on remaining Real
        #real_train = real_train_samples[:n_real_train]
        #real_val = real_val_samples
        
        #train_samples = synthetic_train_samples + real_train
        #val_samples = real_val
        folder_name = f"visual_results_finetune_with{int(real_epochs)}_epochs"
    
    print(f"Syntetic training set size: {len(synthetic_train_samples)}")
    print(f"Real training set size: {len(real_train_samples)}")
    print(f"Validation set size: {len(real_val_samples)}")
    
    # 3. Transforms and Loaders
    transform = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
    ])
    
    synthetic_train_ds = ChessBoardDataset(synthetic_train_samples, transform=transform)
    real_train_ds = ChessBoardDataset(real_train_samples, transform=transform)
    real_val_ds = ChessBoardDataset(real_val_samples, transform=transform)
    
    synthetic_train_loader = DataLoader(synthetic_train_ds, batch_size=batch, shuffle=True, num_workers=4)
    real_train_loader = DataLoader(real_train_ds, batch_size=batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(real_val_ds, batch_size=batch, shuffle=True, num_workers=4)
    
    # 4. Model Initialization
    # Using the custom ChessNet from model.py
    model = ChessNet(num_classes=13, resolution=RESOLUTION).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    max_val_acc = 0.0
    for epoch in tqdm(range(synthetic_epochs + real_epochs)):
        if(epoch >= synthetic_epochs):
            new_train_loader = real_train_loader
        elif add_blur or add_noise:
            new_train_loader = data.generate_augmented_batches_by_photo(add_blur, add_noise, synthetic_train_loader)
        else:
            new_train_loader = synthetic_train_loader

        # Using functions from train_utils.py
        model, train_loss, train_acc = train_one_epoch(model, new_train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if(1 - val_acc) > max_val_acc:
            best_model_wts = model.state_dict()
            max_val_acc = 1 - val_acc
        train_losses.append(train_acc)
        val_losses.append(val_acc)

    
    # 6. Final Evaluation
    #model.load_state_dict(best_model_wts) # Load best model weights from each train epoch
    print(f"Best Validation Miss Rate: {max_val_acc*100:.2f}%")
    print("Training Complete. Evaluating Full Board Accuracy...")
    plot.plot_list(train_losses, "Loss", "Epochs", 
                   f"Training miss rate over Epochs with {real_epochs} real data epochs%", 
                   save_dir=folder_name)
    plot.plot_list(val_losses, "Loss", "Epochs", 
                   f"Validation miss rate over Epochs with {real_epochs} real data epochs%", 
                   save_dir=folder_name)
    evaluate_full_board_accuracy(model, val_loader, device, folder_name=folder_name)
    
    # 7. Save Model Weights (CRITICAL step for predict_board to work)
    MODEL_PATH = 'best_model.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model weights saved successfully to {MODEL_PATH}")

if __name__ == '__main__':
    main()


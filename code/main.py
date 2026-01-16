import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# --- Import custom modules ---
from data import scan_game, ChessBoardDataset
from eval_utils import evaluate_full_board_accuracy
from model import ChessNet  # Importing the STN+CNN model
from train_utils import train_one_epoch, validate # Importing training logic

def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Train ChessNet with STN")
    parser.add_argument('--data_root', type=str, required=True, help='Path to data folder')
    parser.add_argument('--games', nargs='+', required=True, help='List of game folders')
    parser.add_argument('--out', type=str, default='experiments', help='Output folder')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mode', type=str, choices=['zero_shot', 'finetune'], default='zero_shot', help='Training mode')
    parser.add_argument('--real_percent', type=float, default=0.1, help='Fraction of real data to use in train for finetune')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} with mode: {args.mode}")
    
    set_seed()
    
    # 2. Data Preparation
    all_samples = []
    for game in args.games:
        game_path = os.path.join(args.data_root, game)
        # Assuming CSV filename matches game folder name prefix (e.g., game2.csv)
        csv_path = os.path.join(game_path, f"{game.split('_')[0]}.csv")
        print(f"Found CSV for {game}: {csv_path}")
        all_samples.extend(scan_game(game_path, csv_path))
        
    # Split by domain
    synthetic_samples = [s for s in all_samples if s.domain == 'synthetic']
    real_samples = [s for s in all_samples if s.domain == 'real']
    
    # Logic for Training Mode
    if args.mode == 'zero_shot':
        # Train on Synthetic ONLY, Validate on Real
        train_samples = synthetic_samples
        val_samples = real_samples
        folder_name = "visual_results_zero_shot"
        
    elif args.mode == 'finetune':
        # Train on Synthetic + Small % of Real, Validate on remaining Real
        random.shuffle(real_samples)
        n_real_train = int(len(real_samples) * args.real_percent)
        real_train = real_samples[:n_real_train]
        real_val = real_samples[n_real_train:]
        
        train_samples = synthetic_samples + real_train
        val_samples = real_val
        folder_name = f"visual_results_finetune_{int(args.real_percent*100)}"
    
    print(f"Training set: {len(train_samples)} samples")
    print(f"Validation set: {len(val_samples)} samples")
    
    # 3. Transforms and Loaders
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_ds = ChessBoardDataset(train_samples, transform=transform)
    val_ds = ChessBoardDataset(val_samples, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    
    # 4. Model Initialization
    # Using the custom ChessNet from model.py
    model = ChessNet(num_classes=13).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        # Using functions from train_utils.py
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} Tile-Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Tile-Acc: {val_acc:.2f}%")
        print("")
    
    # 6. Final Evaluation
    print("Training Complete. Evaluating Full Board Accuracy...")
    evaluate_full_board_accuracy(model, val_loader, device, folder_name=folder_name)

if __name__ == '__main__':
    main()
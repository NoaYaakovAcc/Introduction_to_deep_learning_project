import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Import our modules
from data import scan_game, ChessBoardDataset
from model import ChessNet
from train_utils import train_one_epoch, validate
from eval_utils import evaluate_full_board_accuracy

def get_transforms(mode="train"):
    """
    Returns the transformation pipeline.
    Note: STN requires a fixed input size (here 256x256).
    
    For training ('train'), we add strong augmentations to synthetic data
    to simulate real-world imperfections (Domain Randomization).
    """
    transforms_list = [transforms.Resize((256, 256))]
    
    if mode == "train":
        transforms_list.extend([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            # RandomPerspective helps the STN learn to be robust to angles
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # For validation, just resize and normalize
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    return transforms.Compose(transforms_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root folder of data")
    parser.add_argument("--games", nargs="+", required=True, help="List of game folders")
    parser.add_argument("--csv_name", type=str, default="game.csv")
    parser.add_argument("--out", type=str, default="experiments")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Lower learning rate for STN stability")
    
    # Sim-to-Real Modes
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "finetune"])
    parser.add_argument("--real_percent", type=float, default=0.1)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    
    print(f"Running on {device} with mode: {args.mode}")
    
    # 1. Data Loading
    all_samples = []
    for g in args.games:
        path = os.path.join(args.data_root, g)
        csv = os.path.join(path, args.csv_name)
        all_samples.extend(scan_game(path, csv))
        
    if not all_samples:
        print("No samples found. Check paths.")
        return

    # Split by domain
    syn_samples = [s for s in all_samples if s.domain == 'synthetic']
    real_samples = [s for s in all_samples if s.domain == 'real']
    
    # Select datasets based on mode
    if args.mode == "zero_shot":
        # Train on Synthetic, Validate on Real
        train_samples = syn_samples
        val_samples = real_samples
    else: # finetune
        # Train on Synthetic + % of Real
        cut = int(len(real_samples) * args.real_percent)
        train_samples = syn_samples + real_samples[:cut]
        val_samples = real_samples[cut:]
        
    print(f"Training set: {len(train_samples)} samples")
    print(f"Validation set: {len(val_samples)} samples")
    
    train_ds = ChessBoardDataset(train_samples, transform=get_transforms("train"))
    val_ds = ChessBoardDataset(val_samples, transform=get_transforms("val"))
    
    # Note: Smaller batch size is safer for full images + STN memory usage
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    # 2. Model Setup
    model = ChessNet(num_classes=13).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    best_acc = 0.0
    for ep in range(args.epochs):
        print(f"\nEpoch {ep+1}/{args.epochs}")
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {tl:.4f} Tile-Acc: {ta:.2%} | Val Loss: {vl:.4f} Tile-Acc: {va:.2%}")
        
        # Save best model based on validation tile accuracy
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pth"))
            
    print("Training Complete. Evaluating Full Board Accuracy (Strict Metric)...")
    
    # Final Evaluation
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pth")))
    evaluate_full_board_accuracy(model, val_loader, device)

if __name__ == "__main__":
    main()
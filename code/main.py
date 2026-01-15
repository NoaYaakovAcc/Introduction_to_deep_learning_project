import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Import custom modules
from data import scan_game, ChessBoardDataset
from eval_utils import evaluate_full_board_accuracy

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(device):
    # Using ResNet18 as a backbone
    model = models.resnet18(pretrained=True)
    # Modify the first layer to accept the specific input size if needed, 
    # but ResNet takes standard images so we just modify the output.
    
    # We need 64 outputs, each with 13 classes (12 pieces + empty)
    # ResNet18 output is 512 before the FC layer
    model.fc = nn.Linear(512, 64 * 13)
    return model.to(device)

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_tiles = 0
        total_tiles = 0
        
        for imgs, labels, _ in train_loader: # Unpack 3 values (ignore path)
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(imgs) # [B, 64*13]
            
            # Reshape output to [B, 64, 13] for CrossEntropy
            output = output.view(-1, 64, 13)
            
            # CrossEntropy expects [B, Class, D1, D2...] or flat [N, Class]
            # Flatten to [B*64, 13] and labels to [B*64]
            loss = criterion(output.view(-1, 13), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate simple tile accuracy
            preds = output.argmax(dim=2)
            correct_tiles += (preds == labels).sum().item()
            total_tiles += labels.numel()
            
        avg_loss = total_loss / len(train_loader)
        tile_acc = 100.0 * correct_tiles / total_tiles
        
        # Validation step
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f} Tile-Acc: {tile_acc:.2f}% | Val Loss: {val_loss:.4f} Tile-Acc: {val_acc:.2f}%")
        print("")

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels, _ in val_loader: # Unpack 3 values
            imgs, labels = imgs.to(device), labels.to(device)
            
            output = model(imgs)
            output = output.view(-1, 64, 13)
            loss = criterion(output.view(-1, 13), labels.view(-1))
            
            total_loss += loss.item()
            preds = output.argmax(dim=2)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            
    return total_loss / len(val_loader), 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to data folder')
    parser.add_argument('--games', nargs='+', required=True, help='List of game folders')
    parser.add_argument('--out', type=str, default='experiments')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mode', type=str, choices=['zero_shot', 'finetune'], default='zero_shot')
    parser.add_argument('--real_percent', type=float, default=0.1, help='Fraction of real data to use in train for finetune')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} with mode: {args.mode}")
    
    set_seed()
    
    # 1. Collect all samples
    all_samples = []
    for game in args.games:
        game_path = os.path.join(args.data_root, game)
        csv_path = os.path.join(game_path, f"{game.split('_')[0]}.csv") # e.g. game2.csv
        print(f"Found CSV for {game}: {csv_path}")
        all_samples.extend(scan_game(game_path, csv_path))
        
    # 2. Split into Synthetic and Real
    synthetic_samples = [s for s in all_samples if s.domain == 'synthetic']
    real_samples = [s for s in all_samples if s.domain == 'real']
    
    # 3. Create Train/Val Split based on Mode
    if args.mode == 'zero_shot':
        # Train on ALL synthetic, Val on ALL real
        train_samples = synthetic_samples
        val_samples = real_samples
        
    elif args.mode == 'finetune':
        # Shuffle real samples
        random.shuffle(real_samples)
        
        # Take a percentage of real samples for training
        n_real_train = int(len(real_samples) * args.real_percent)
        real_train = real_samples[:n_real_train]
        real_val = real_samples[n_real_train:]
        
        # Train = All Synthetic + Some Real
        train_samples = synthetic_samples + real_train
        val_samples = real_val
    
    print(f"Training set: {len(train_samples)} samples")
    print(f"Validation set: {len(val_samples)} samples")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_ds = ChessBoardDataset(train_samples, transform=transform)
    val_ds = ChessBoardDataset(val_samples, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    
    # Model Setup
    model = get_model(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    train(model, train_loader, val_loader, optimizer, criterion, device, args.epochs)
    
    # Final Evaluation
    print("Training Complete. Evaluating Full Board Accuracy (Strict Metric)...")
    evaluate_full_board_accuracy(model, val_loader, device)

if __name__ == '__main__':
    main()
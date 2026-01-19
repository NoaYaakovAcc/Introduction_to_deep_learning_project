import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

# --- Import custom modules ---
from data import scan_game, ChessBoardDataset
from eval_utils import evaluate_full_board_accuracy
from model import ChessNet
from train_utils import train_one_epoch, validate

# --- GLOBAL CONFIGURATION FOR INFERENCE ---
# The requirements specify the output tensor must be on CPU.
# We also define the path where the model weights will be saved/loaded from.
PREDICT_DEVICE = torch.device('cpu') 
MODEL_PATH = 'best_model.pth'

def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# REQUIRED EVALUATION FUNCTION (PROJECT 2)
# ==========================================
def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Mandatory evaluation function.
    
    Args:
        image (np.ndarray): Input image with shape (H, W, 3), RGB, uint8.
        
    Returns:
        torch.Tensor: A (8, 8) tensor on CPU containing class indices (int64).
    """
    # 1. Initialize Model Architecture
    # We must initialize the same architecture used during training.
    model = ChessNet(num_classes=13)
    
    # 2. Load Trained Weights
    # We use map_location='cpu' to ensure strict compliance with the CPU requirement,
    # and to allow evaluation on machines without GPUs.
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=PREDICT_DEVICE))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            # In a real scenario, you might want to raise an error here.
    else:
        print(f"Warning: {MODEL_PATH} not found. Ensure you have trained the model first.")

    # Move model to CPU and set to evaluation mode
    model.to(PREDICT_DEVICE)
    model.eval()

    # 3. Preprocessing
    # Must match the training transformations exactly.
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Convert Numpy (uint8) -> PIL Image -> Tensor
    pil_img = Image.fromarray(image.astype('uint8')).convert('RGB')
    img_tensor = transform_pipeline(pil_img)
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(PREDICT_DEVICE)

    # 4. Inference
    with torch.no_grad():
        # Forward pass. Output shape: [1, 64, 13]
        logits = model(img_tensor)
        
        # Get class predictions (argmax). Shape: [1, 64]
        preds = torch.argmax(logits, dim=2)
        
        # Reshape to 8x8 grid as required by the spec
        board_output = preds.view(8, 8)
        
    # 5. Return requirements: strictly CPU tensor, int64 dtype
    return board_output.cpu().long()

# ==========================================
# MAIN TRAINING SCRIPT
# ==========================================
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
    
    # Use GPU for training if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running training on {device} with mode: {args.mode}")
    
    set_seed()
    
    # 2. Data Preparation
    all_samples = []
    for game in args.games:
        game_path = os.path.join(args.data_root, game)
        # Construct CSV path assuming standard naming convention
        csv_path = os.path.join(game_path, f"{game.split('_')[0]}.csv")
        print(f"Looking for data in: {game_path} | CSV: {csv_path}")
        all_samples.extend(scan_game(game_path, csv_path))
        
    # Split by domain (Synthetic vs Real)
    synthetic_samples = [s for s in all_samples if s.domain == 'synthetic']
    real_samples = [s for s in all_samples if s.domain == 'real']
    
    # Configure Datasets based on Mode
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
    
    print(f"Training set size: {len(train_samples)}")
    print(f"Validation set size: {len(val_samples)}")
    
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
    model = ChessNet(num_classes=13).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    print("Starting training process...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    
    # 6. Final Evaluation & Visualization
    print("Evaluating Full Board Accuracy on Validation Set...")
    evaluate_full_board_accuracy(model, val_loader, device, folder_name=folder_name)
    
    # 7. Save Model Weights (CRITICAL step for predict_board to work)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model weights saved successfully to {MODEL_PATH}")

if __name__ == '__main__':
    main()
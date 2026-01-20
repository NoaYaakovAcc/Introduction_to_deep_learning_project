import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur

def plot_noisy_image(image_path):
    # Load and transform to tensor (C, H, W)
    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img)

    # Add noise and clamp to valid range [0, 1]
    noisy_tensor = add_blur(img_tensor, kernel_size=9, sigma=3.0)
    noisy_tensor = add_noise(noisy_tensor, std=0.1)
    noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

    # Prepare for plotting (H, W, C)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    noisy_np = noisy_tensor.permute(1, 2, 0).numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(noisy_np)
    axes[1].set_title("Noisy")
    axes[1].axis("off")
    
    plt.show()

IMG_SIZE = (480, 480)

def add_blur(tensor, kernel_size=9, sigma=3.0):
    """
    Applies Gaussian Blur to a tensor.
    kernel_size must be an odd number (e.g., 5, 9).
    Higher sigma means more blur.
    """
    transform = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return transform(tensor)

def add_noise(tensor, std=0.05):
    """
    Adds Gaussian noise to a tensor.
    The std parameter controls intensity.
    """
    return tensor + torch.randn_like(tensor) * std


class ChessBoardSample:
    def __init__(self, img_path, fen, domain):
        self.img_path = img_path
        self.fen = fen
        self.domain = domain

def infer_domain(path):
    # Determines if image is synthetic based on folder name
    return "synthetic" if "generated" in path.lower() else "real"

def scan_game(game_root, csv_path):
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}")
        return []
    
    try:
        df = pd.read_csv(csv_path)
        # Strip whitespace from columns to avoid key errors
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return []

    # Map all image files
    all_files = glob.glob(os.path.join(game_root, "**", "*.jpg"), recursive=True)
    all_files += glob.glob(os.path.join(game_root, "**", "*.png"), recursive=True)

    samples = []
    
    for _, row in df.iterrows():
        try:
            fen = row['fen'] 
            frame_num = int(row['from_frame'])
            # Assuming 6-digit filename format 
            # example: frame_000200.jpg
            filename = f"frame_{frame_num:06d}.jpg" 
        except (KeyError, ValueError):
            continue
        fen_files = [file for file in all_files if filename in file]
        for full_path in fen_files:
            samples.append(ChessBoardSample(full_path, fen, infer_domain(full_path)))
            
    print(f"  [Scan] Found {len(samples)} valid samples in {game_root}")
    return samples

class ChessBoardDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.piece_map = {
            # White pieces
            'P': 0,  # Pawn
            'R': 1,  # Rook 
            'N': 2,  # Knight 
            'B': 3,  # Bishop 
            'Q': 4,  # Queen
            'K': 5,  # King
            
            # Black pieces
            'p': 6,  # Pawn
            'r': 7,  # Rook 
            'n': 8,  # Knight 
            'b': 9,  # Bishop 
            'q': 10, # Queen
            'k': 11, # King
            
            # Empty
            '.': 12
        }

    def parse_fen(self, fen):
        board_str = fen.split()[0]
        rows = board_str.split('/')
        labels = []
        for r in rows:
            for c in r:
                if c.isdigit():
                    labels.extend([12] * int(c)) 
                else:
                    labels.append(self.piece_map.get(c, 12)) 
        
        # Ensure length is 64
        if len(labels) != 64:
             labels = labels[:64] + [12] * (64 - len(labels))

        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s.img_path).convert('RGB')
            labels = self.parse_fen(s.fen)
            
            if self.transform:
                img = self.transform(img)
            
            # Return image, label, AND path (for visualization)
            return img, labels, s.img_path
        except Exception as e:
            print(f"Error loading {s.img_path}: {e}")
            return torch.zeros(3, *IMG_SIZE), torch.zeros(64, dtype=torch.long), "error"
        
#plot_noisy_image(r"C:\Users\yoavl\Documents\github\Introduction_to_deep_learning_project\data\game2_per_frame\tagged_images\frame_000200.jpg")
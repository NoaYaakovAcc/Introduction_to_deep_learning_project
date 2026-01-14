import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Constant for image size. STN requires a fixed input size to learn the affine matrix.
IMG_SIZE = (256, 256)

class ChessBoardSample:
    """
    Stores metadata for a single board sample.
    """
    def __init__(self, img_path, fen, domain):
        self.img_path = img_path
        self.fen = fen
        self.domain = domain

def infer_domain(path):
    """
    Helper to determine domain based on folder structure.
    'GENERATED' -> Synthetic (Blender)
    'TAGGED' -> Real (Camera)
    """
    return "synthetic" if "GENERATED" in path else "real"

def scan_game(game_root, csv_path):
    """
    Scans a game directory and matches images to their FEN labels via the CSV.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}")
        return []
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return []

    # Find all images recursively
    all_imgs = glob.glob(os.path.join(game_root, "**", "*.jpg"), recursive=True)
    all_imgs += glob.glob(os.path.join(game_root, "**", "*.png"), recursive=True)
    
    # Create a map for fast lookup: filename -> full path
    img_map = {os.path.basename(p): p for p in all_imgs}
    
    samples = []
    for _, row in df.iterrows():
        try:
            # Column names based on your project description
            fen = row['frame fen']
            fid = str(row['from frame']).strip()
        except KeyError:
            continue
        
        # Try to find the image corresponding to this frame ID
        path = None
        for k, v in img_map.items():
            if fid in k: 
                path = v
                break
        
        if path:
            samples.append(ChessBoardSample(path, fen, infer_domain(path)))
            
    return samples

class ChessBoardDataset(Dataset):
    """
    PyTorch Dataset that returns:
    1. Full Board Image (resized)
    2. Vector of 64 labels (integers)
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
        # Map piece characters to class indices (0-12)
        self.piece_map = {
            'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,  # White
            'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11, # Black
            '.':12                                      # Empty
        }

    def parse_fen(self, fen):
        """
        Parses a FEN string into a tensor of 64 class indices.
        """
        board = fen.split()[0]
        rows = board.split('/')
        labels = []
        for r in rows:
            for c in r:
                if c.isdigit():
                    # FEN numbers represent consecutive empty spaces
                    labels.extend([12] * int(c)) 
                else:
                    labels.append(self.piece_map[c])
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s.img_path).convert('RGB')
            labels = self.parse_fen(s.fen) # Tensor of shape [64]
            
            if self.transform:
                img = self.transform(img)
                
            return img, labels
        except Exception as e:
            print(f"Error loading {s.img_path}: {e}")
            # Return dummy data on error to prevent training crash
            return torch.zeros(3, *IMG_SIZE), torch.zeros(64, dtype=torch.long)
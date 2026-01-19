import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_SIZE = (480, 480)

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
    
    img_map = {os.path.basename(f): f for f in all_files}
    samples = []
    
    for _, row in df.iterrows():
        try:
            fen = row['fen'] 
            frame_num = int(row['from_frame'])
            # Assuming 6-digit filename format e.g. frame_000200.jpg
            filename = f"frame_{frame_num:06d}.jpg" 
        except (KeyError, ValueError):
            continue

        if filename in img_map:
            full_path = img_map[filename]
            samples.append(ChessBoardSample(full_path, fen, infer_domain(full_path)))
            
    print(f"  [Scan] Found {len(samples)} valid samples in {game_root}")
    return samples

class ChessBoardDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.piece_map = {
            'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,
            'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11,
            '.':12
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
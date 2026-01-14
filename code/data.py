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
    If the path contains 'generated', it is synthetic data.
    Otherwise, it is real (tagged) data.
    """
    # Case-insensitive check for 'generated' in the file path
    return "synthetic" if "generated" in path.lower() else "real"

def scan_game(game_root, csv_path):
    """
    Scans a game directory and matches images to their FEN labels via the CSV.
    
    Logic:
    1. Read the CSV to get frame numbers and FEN strings.
    2. Recursively find all image files (.jpg, .png) in the game folder.
    3. Match the CSV frame number to the image filename using 6-digit format.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}")
        return []
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        # Strip whitespace from column names to avoid key errors (e.g., ' fen' vs 'fen')
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return []

    # 1. Map all images in the directory (both generated and tagged)
    # We create a dictionary: filename -> full_path
    # Example: 'frame_000200.jpg' -> '/home/.../data/game2/generated_images/frame_000200.jpg'
    all_files = glob.glob(os.path.join(game_root, "**", "*.jpg"), recursive=True)
    all_files += glob.glob(os.path.join(game_root, "**", "*.png"), recursive=True)
    
    img_map = {os.path.basename(f): f for f in all_files}
    
    samples = []
    
    # 2. Iterate over the CSV and find matching images
    for _, row in df.iterrows():
        try:
            # Extract data from the row
            # Note: These keys must match your CSV headers exactly (from_frame, fen)
            fen = row['fen'] 
            frame_num = int(row['from_frame']) # e.g., 200
            
            # Construct the expected filename with 6 digits (e.g., frame_000200.jpg)
            filename = f"frame_{frame_num:06d}.jpg" 
            
        except KeyError as e:
            # Skip if a column is missing
            continue
        except ValueError:
            # Skip if frame number is not an integer
            continue

        # Check if this filename exists in our found images
        if filename in img_map:
            full_path = img_map[filename]
            domain = infer_domain(full_path)
            samples.append(ChessBoardSample(full_path, fen, domain))
            
    # Print a summary for debugging purposes
    print(f"  [Scan] Found {len(samples)} valid samples in {game_root}")
            
    return samples

class ChessBoardDataset(Dataset):
    """
    PyTorch Dataset that returns:
    1. Full Board Image (resized)
    2. Vector of 64 labels (integers) representing the board state.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
        # Map piece characters to class indices (0-12)
        self.piece_map = {
            'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,   # White Pieces
            'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11, # Black Pieces
            '.':12                                      # Empty Square
        }

    def parse_fen(self, fen):
        """
        Parses a FEN string into a tensor of 64 class indices.
        Example FEN: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        """
        # We only need the board position (the first part of the string)
        board_str = fen.split()[0]
        rows = board_str.split('/')
        labels = []
        
        for r in rows:
            for c in r:
                if c.isdigit():
                    # Digits in FEN represent consecutive empty spaces
                    labels.extend([12] * int(c)) 
                else:
                    # Letters represent pieces. Default to Empty (12) if unknown char.
                    labels.append(self.piece_map.get(c, 12)) 
        
        # Ensure the vector is exactly length 64
        if len(labels) != 64:
             # Pad with empty squares or truncate if necessary
             labels = labels[:64] + [12] * (64 - len(labels))

        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            # Load image and convert to RGB (removes alpha channel if present)
            img = Image.open(s.img_path).convert('RGB')
            labels = self.parse_fen(s.fen) # Tensor of shape [64]
            
            if self.transform:
                img = self.transform(img)
                
            return img, labels
        except Exception as e:
            print(f"Error loading {s.img_path}: {e}")
            # Return dummy data on error to prevent training crash
            return torch.zeros(3, *IMG_SIZE), torch.zeros(64, dtype=torch.long)
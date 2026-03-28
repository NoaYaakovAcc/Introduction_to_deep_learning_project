import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt



IMG_SIZE = (480, 480)


def collate_with_paths(batch):
    images = torch.stack([sample[0] for sample in batch])
    labels = torch.stack([sample[1] for sample in batch])
    image_paths = [sample[2] for sample in batch]
    return images, labels, image_paths

class AugmentedDataset(Dataset):
    def __init__(self, images, labels, image_paths):
        self.images = images
        self.labels = labels
        self.image_paths = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.image_paths[index]

def build_augmented_batches_loader(train_loader):

    augmented_images = []
    augmented_labels = []
    augmented_paths = []

    for batch_images, batch_labels, batch_paths in train_loader:
        batch_images = add_noise(batch_images)

        for i in range(batch_images.shape[0]):
            augmented_images.append(batch_images[i])
            augmented_labels.append(batch_labels[i])
            augmented_paths.append(batch_paths[i])

    augmented_dataset = AugmentedDataset(
        augmented_images,
        augmented_labels,
        augmented_paths)

    return DataLoader(
        augmented_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_with_paths
)

def plot_noisy_image(image_path): # do we need to have it in the subbmision 
    image = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)

    noisy_tensor = add_noise(image_tensor.unsqueeze(0)).squeeze(0)

    original_image = image_tensor.permute(1, 2, 0).numpy()
    noisy_image = noisy_tensor.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy_image)
    axes[1].set_title("Noisy")
    axes[1].axis("off")

    plt.show()

def add_noise(images, std=0.1):
    """
    Adds Gaussian noise to a tensor.
    The std parameter controls intensity.
    """
    noisy_images = images + torch.randn_like(images) * std
    return torch.clamp(noisy_images, 0, 1)


class ChessBoardSample:
    def __init__(self, image_path, fen, domain):
        self.image_path = image_path
        self.fen = fen
        self.domain = domain

def infer_domain(image_path):
    # Determines if image is synthetic based on folder name
    return "synthetic" if "generated" in image_path.lower() else "real"

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
    all_image_files = glob.glob(os.path.join(game_root, "**", "*.jpg"), recursive=True)
    all_image_files += glob.glob(os.path.join(game_root, "**", "*.png"), recursive=True)

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
        
        matching_files = [file_path for file_path in all_image_files if filename in file_path]

        for full_path in matching_files:
            samples.append(
                ChessBoardSample(
                    full_path,
                    fen,
                    infer_domain(full_path)
                )
            )
            
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

        for row in rows:
            for char in row:
                if char.isdigit():
                    labels.extend([12] * int(char))
                else:
                    labels.append(self.piece_map.get(char, 12))
        
        # Ensure length is 64
        if len(labels) != 64:
             labels = labels[:64] + [12] * (64 - len(labels))

        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            image = Image.open(sample.image_path).convert("RGB")
            labels = self.parse_fen(sample.fen)

            if self.transform is not None:
                image = self.transform(image)

            return image, labels, sample.image_path

        except Exception as error:
            print(f"Error loading {sample.image_path}: {error}")
            return (
                torch.zeros(3, *IMG_SIZE),
                torch.zeros(64, dtype=torch.long),
                "error"
            )
        
#plot_noisy_image(r"C:\Users\yoavl\Documents\github\Introduction_to_deep_learning_project\data\game2_per_frame\tagged_images\frame_000200.jpg")
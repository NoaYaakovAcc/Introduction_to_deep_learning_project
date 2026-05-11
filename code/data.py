import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt



IMAGE_SIZE = (480, 480)


def collate_with_paths(batch):
    """
    Custom collate function for PyTorch DataLoader.

    Parameters:
        batch : list[tuple] -> List of samples returned by the dataset.
                                Each sample: (image_tensor, label_tensor, image_path)
        
    Returns:
        tuple -> consisting of: images : torch.Tensor -> Image tensor of shape [batch_size, C, H, W]
                                labels : torch.Tensor -> Label tensor of shape [batch_size, 64]
                                image_paths : list[str] -> List of image path strings for the batch
                
    """
    images = torch.stack([sample[0] for sample in batch])
    labels = torch.stack([sample[1] for sample in batch])
    image_paths = [sample[2] for sample in batch]
    return images, labels, image_paths

class AugmentedDataset(Dataset):
    """
    This class is used after generating augmented images from an existing loader

    Attributes:
        images : list[torch.Tensor] -> List of augmented image tensors
        labels : list[torch.Tensor] -> List of label tensors
        image_paths : list[str] -> List of image path strings

    """
    def __init__(self, images, labels, image_paths):
        self.images = images
        self.labels = labels
        self.image_paths = image_paths

    def __len__(self):
        """
        Return the number of samples in the augmented dataset.

        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Return one augmented sample.

        """
        return self.images[index], self.labels[index], self.image_paths[index]

def build_augmented_batches_loader(train_loader):
    """
    Builds a new DataLoader with noisy versions of the training images.

    Parameters:
        train_loader : torch.utils.data.DataLoader -> Original training loader
        

    Returns:
        torch.utils.data.DataLoader -> New dataloader with added noise 
        
    """

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
    """
    Display a side-by-side comparison of an original image and its noisy version.

    Parameters:
        image_path : string -> Path to the image file to visualize
        
    Returns:
        None
    """
    image = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)

    #adding noise
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

    Parameters:
        images : torch.Tensor -> Input tensor of images.
        std : float -> Standard deviation of Gaussian noise.

    Returns:
        torch.Tensor -> Noisy image tensor, in range [0, 1]
       
    """
    noisy_images = images + torch.randn_like(images) * std
    return torch.clamp(noisy_images, 0, 1)


class ChessBoardSample:
    """
    representing one chessboard sample.

    Attributes:
        image_path : string -> Full path to the image file
        fen : string -> Board state encoded as a FEN string
        domain : string -> Domain label, 'synthetic' or 'real' data
        
    """
    def __init__(self, image_path, fen, domain):
        self.image_path = image_path
        self.fen = fen
        self.domain = domain

def infer_domain(image_path):
    """
    Determines if image is synthetic based on folder name

    Parameters:
        image_path : string -> Path to the image
        
    Returns:
        string -> 'synthetic' or 'real'
        
    """
    return "synthetic" if "generated" in image_path.lower() else "real"

def scan_game(game_root, csv_path):
    """
    Scan one game directory and create a list of labeled samples.

    Parameters:
        game_root : string -> Root directory of the game folder
        csv_path : string -> Path to the CSV annotation file
        
    Returns:
        list[ChessBoardSample] -> List of valid samples (matched correctly)
        
    Expected CSV columns include:
    - 'fen'
    - 'from_frame'

    
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}")
        return []
    
    try:
        # Read the annotation CSV
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
    
    # Search recursively for image files
    for _, row in df.iterrows():
        try:
            fen = row['fen'] 
            frame_num = int(row['from_frame'])
            # Expected image naming format:
            # e.g, frame_000200.jpg
            filename = f"frame_{frame_num:06d}.jpg" 
        except (KeyError, ValueError):
            continue
        
        # matches frame numbers from the CSV to image filenames
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
    """
    Main PyTorch dataset for chessboard square classification.

    

    Attributes:
        samples : list[ChessBoardSample] -> List of sample objects
                                            Each sample consists of: - one RGB board image
                                                                     - one label tensor of length 64
                                                                     - the image path
        transform : callable, optional -> image transform pipeline
        
    """
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
        """
        Convert a fen board string into a tensor of 64 class labels.

        Parameters:
            fen : strimg -> Fen board string

        Returns:
            torch.Tensor -> Tensor of shape [64] 
            
        If the resulting board length is not exactly 64, the output is
        padded or truncated to length 64.
        """
         
        board_str = fen.split()[0]
        rows = board_str.split('/')

        labels = []

        for row in rows:
            for char in row:
                if char.isdigit(): 
                    #Empty squares in a row
                    labels.extend([12] * int(char))
                else:
                    labels.append(self.piece_map.get(char, 12))
        
        # Ensure length is 64
        if len(labels) != 64:
             labels = labels[:64] + [12] * (64 - len(labels))

        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Load one sample from the dataset.
        One sample: tuple -> consisting of: image : torch.Tensor -> Transformed image tensor
                                            labels : torch.Tensor -> Label tensor of shape [64]
                                            image_path : string -> image path    

        """
        sample = self.samples[index]
        try:
            image = Image.open(sample.image_path).convert("RGB")
            labels = self.parse_fen(sample.fen)

            if self.transform is not None:
                image = self.transform(image)

            return image, labels, sample.image_path

        #If loading fails, a fallback sample is returned: path string "error", zero image and labels
        except Exception as error:
            print(f"Error loading {sample.image_path}: {error}")
            return (
                torch.zeros(3, *IMAGE_SIZE),
                torch.zeros(64, dtype=torch.long),
                "error"
            )
        
#plot_noisy_image(r"C:\Users\yoavl\Documents\github\Introduction_to_deep_learning_project\data\game2_per_frame\tagged_images\frame_000200.jpg")
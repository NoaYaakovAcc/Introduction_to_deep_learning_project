import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    """
    Spatial Transformer Network (STN).
    Learns to estimate an affine transformation matrix (theta) to rectifty the input image.
    Reference: 'Intro_to_STN (2).pdf' from course materials.
    """
    def __init__(self):
        super(STN, self).__init__()
        
        # 1. Localization Network: Extracts features to predict transformation parameters
        self.loc_net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3x2 affine matrix.
        # We use AdaptiveAvgPool to ensure fixed input size for the Linear layer
        # regardless of the input image dimensions.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2) # Outputs 6 parameters for the affine matrix
        )
        
        # Initialize with Identity transformation (no distortion)
        # This helps the model start training from a stable state.
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = self.adaptive_pool(xs)
        xs = xs.view(-1, 10 * 4 * 4)
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3) # Shape: [Batch, 2, 3]
        
        # 2. Grid Generator and Sampler
        # Creates a grid based on theta and samples pixels from input x
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_transformed = F.grid_sample(x, grid, align_corners=True)
        
        return x_transformed

class ChessNet(nn.Module):
    """
    End-to-End Network: STN -> Grid Slicing -> Classification.
    """
    def __init__(self, num_classes=13, resolution=480):
        super(ChessNet, self).__init__()
        self.stn = STN()
        self.resolution = resolution
        self.tile_size = resolution // 8
        
        # Tile Classifier: A simple CNN that processes a single tile_size x tile_size tile.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2), # tile_size -> tile_size//2
            
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2), # tile_size//2 -> tile_size//4
            
            nn.Flatten()
        )
        
        # Input features: 64 channels * (tile_size//4) * (tile_size//4) spatial size
        self.fc = nn.Linear(64 * (self.tile_size // 4) * (self.tile_size // 4), num_classes)

    def forward(self, x):
        # 1. Apply STN to rectify the full board image
        x = self.stn(x) 
        
        # 2. Slice the rectified image into 64 tiles using tensor operations
        # Input x shape: [Batch, 3, resolution, resolution]
        B, C, H, W = x.shape
        h, w = self.tile_size, self.tile_size  # Tile size (e.g., 32 for 256x256)
        
        # 'unfold' extracts sliding local blocks.
        # We unfold height (dim 2) and width (dim 3).
        tiles = x.unfold(2, h, h).unfold(3, w, w) 
        # tiles shape: [B, C, 8, 8, h, w]
        
        # Permute to put the grid dimensions (8,8) together
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
        # Shape: [B, 8, 8, C, h, w]
        
        # Flatten the grid to treat each tile as a separate sample in the batch
        # New shape: [B * 64, C, h, w]
        tiles = tiles.view(-1, C, h, w)
        
        
        # 3. Classify all tiles simultaneously
        features = self.conv(tiles)
        logits = self.fc(features)
        
        # Reshape back to [Batch, 64, NumClasses]
        return logits.view(B, 64, 13)
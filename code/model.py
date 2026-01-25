import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
    def __init__(self, num_classes=13, resolution=480 ,expansion_ratio = 1.3):
        super(ChessNet, self).__init__()
        self.stn = STN()
        self.resolution = resolution
        self.base_tile_size = resolution // 8

        self.expansion_tile_size = int(self.base_tile_size * expansion_ratio)
        self.padding_amount = (self.expansion_tile_size - self.base_tile_size) // 2
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone.maxpool = nn.Identity()

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        

    def forward(self, x):
        # 1. Apply STN to rectify the full board image
        x = self.stn(x) 
        
        
        x_padded = F.pad(x, (self.padding_amount, self.padding_amount, self.padding_amount, self.padding_amount))
        
        kernel = self.expansion_tile_size
        stride = self.base_tile_size
        
        tiles = x_padded.unfold(2, kernel, stride).unfold(3, kernel, stride)
        
        # Permute to put the grid dimensions (8,8) together
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        
        # Flatten the grid to treat each tile as a separate sample in the batch
        # New shape: [B * 64, C, h, w]
        tiles = tiles.view(-1, 3, kernel, kernel)
        
        
        logits = self.backbone(tiles)
        
        # Reshape back to [Batch, 64, NumClasses]
        return logits.view(x.shape[0], 64, 13)
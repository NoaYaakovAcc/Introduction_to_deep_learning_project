import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

NUMBER_OF_CHESS_CLASSES = 13
DEFAULT_RESOLUTION = 480
EXPANTION_RATIO = 1.3
RESNET_VERSION =  18
class Classifier(nn.Module):
    """ResNet18-based feature extractor and classifier."""
    
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.maxpool = nn.Identity()
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ChessNet(nn.Module):
    """End-to-End Chess Board Classifier. Slices board into tiles and classifies pieces."""
    
    def __init__(self, num_classes=NUMBER_OF_CHESS_CLASSES, resolution=DEFAULT_RESOLUTION, expansion_ratio=EXPANTION_RATIO):
        super().__init__()
        self.resolution = resolution
        self.base_tile_size = resolution // 8
        self.expansion_tile_size = int(self.base_tile_size * expansion_ratio)
        self.padding_amount = (self.expansion_tile_size - self.base_tile_size) // 2

        # Type classifier (empty / white / black)
        self.type_classifier = Classifier(num_classes=3)

        # Separate color classifiers for white/black piece type
        self.white_piece_classifier = Classifier(num_classes=6)
        self.black_piece_classifier = Classifier(num_classes=6)

    def forward(self, x):
        B = x.shape[0]
        pad = self.padding_amount
        
        # Pad the image to allow for expanded tile extraction.
        x_padded = F.pad(x, (pad, pad, pad, pad))

        kernel = self.expansion_tile_size
        stride = self.base_tile_size

        # Extract overlapping patches over the 8x8 grid.
        tiles = x_padded.unfold(2, kernel, stride).unfold(3, kernel, stride)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        # Flatten the grid to treat each tile as a separate sample.
        tiles = tiles.view(B * 64, 3, kernel, kernel)

        # 1. Forward passes for all tiles (required for gradients to flow)
        type_logits = self.type_classifier(tiles)
        white_logits = self.white_piece_classifier(tiles)
        black_logits = self.black_piece_classifier(tiles)

        # 2. Convert to probabilities.
        type_probs = F.softmax(type_logits, dim=1)
        white_probs = F.softmax(white_logits, dim=1)
        black_probs = F.softmax(black_logits, dim=1)

        # 3. Isolate the gates.
        empty_prob = type_probs[:, 0:1]  # Index 0 is empty
        white_gate = type_probs[:, 1:2]  # Index 1 is white
        black_gate = type_probs[:, 2:3]  # Index 2 is black

        # 4. Combine into 13-way probabilities.
        # Order must match: White (0-5), Black (6-11), Empty (12)
        combined_probs = torch.cat([
            white_gate * white_probs,  
            black_gate * black_probs,  
            empty_prob                 
        ], dim=1)

        # 5. Convert back to log-space for CrossEntropyLoss.
        combined_logits = torch.log(combined_probs.clamp(min=1e-8))

        return combined_logits.view(B, 64, 13)
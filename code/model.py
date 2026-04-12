import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

NUMBER_OF_CHESS_CLASSES = 13
DEFAULT_RESOLUTION = 480
EXPANTION_RATIO = 1.3
RESNET_VERSION =  18
class ChessNetCheck(nn.Module):
    """
    Chessboard classification network.

    The model receives a full-board image, splits it into 64 tiles
    (one per square), classifies each tile
    using a pretrained ResNet backbone, and returns predictions for all
    64 board positions.

    Output shape:
        [batch_size, 64, num_classes]
    """

    def __init__(
        self,
        num_classes: int = NUMBER_OF_CHESS_CLASSES,
        resolution: int = DEFAULT_RESOLUTION,
        expansion_ratio: float = EXPANTION_RATIO,
        resnet_version: int = RESNET_VERSION,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.resolution = resolution
        self.base_tile_size = resolution // 8

        self.expansion_tile_size = int(self.base_tile_size * expansion_ratio)
        self.padding_amount = (self.expansion_tile_size - self.base_tile_size) // 2

        if resnet_version == 18:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_version == 50:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported resnet_version: {resnet_version}")

        # Avoid too much early downsampling for small tiles
        self.backbone.maxpool = nn.Identity()

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            x : torch.Tensor -> Input tensor of shape [B, 3, H, W]

        Returns:
            torch.Tensor -> Tensor of shape [B, 64, num_classes]

        """
            
        x_padded = F.pad(
            x,
            (
                self.padding_amount,
                self.padding_amount,
                self.padding_amount,
                self.padding_amount,
            ),
        )

        kernel = self.expansion_tile_size
        stride = self.base_tile_size

        # Extract 8x8 tile grid from padded board image
        tiles = x_padded.unfold(2, kernel, stride).unfold(3, kernel, stride)

        # Reorder to [B, 8, 8, C, H, W]
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()

        # Flatten to [B * 64, C, H, W]
        tiles = tiles.view(-1, 3, kernel, kernel)

        # Classify each tile
        logits = self.backbone(tiles)

        # Reshape back to [B, 64, num_classes]
        return logits.view(x.shape[0], 64, self.num_classes)
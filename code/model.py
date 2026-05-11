import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

def get_vision_model(model_name, num_classes=13):
    """
    Loads a vision model by string name.
    Adjusts the final layer to match the classes.
    """
    try:
        if model_name == "inception_v3":
            model = models.get_model(model_name, weights="DEFAULT", aux_logits=False)
        else:
            model = models.get_model(model_name, weights="DEFAULT")
    except ValueError:
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            if isinstance(model.classifier[-1], nn.Linear):
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            elif "squeezenet" in model_name:
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        elif isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "heads") and hasattr(model.heads, "head"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)

    return model
class ChessNet(nn.Module):
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
            num_classes: int = 13,
            resolution: int = 480,
            expansion_ratio: float = 1.3,
            model_name: str = "vgg16",
        ):
            super().__init__()

            self.num_classes = num_classes
            self.resolution = resolution
            self.base_tile_size = resolution // 8

            self.expansion_tile_size = int(self.base_tile_size * expansion_ratio)
            self.padding_amount = (self.expansion_tile_size - self.base_tile_size) // 2

            # This already adapts the final layer correctly.
            self.backbone = get_vision_model(model_name, num_classes=num_classes)

            # Remove maxpool if the model has it.
            if hasattr(self.backbone, "maxpool"):
                self.backbone.maxpool = nn.Identity()

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
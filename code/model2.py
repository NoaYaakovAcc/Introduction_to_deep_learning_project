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

    def __init__(self, num_classes=13, model_name="vgg16"):
        super().__init__()
        self.num_classes = num_classes
        self.num_squares = 64
        
        # Timm automatically replaces the final classification layer.
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=self.num_squares * self.num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the full board.
        Output shape: [B, 64, num_classes].
        """
        logits = self.backbone(x)
        return logits.view(-1, self.num_squares, self.num_classes)
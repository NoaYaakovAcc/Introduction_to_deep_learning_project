import torch.nn as nn

class TileCNN(nn.Module):
    """
    Simple CNN for 3x3-tile classification into 13 classes.
    Input: [B,3,H,W] (after resize to tile_size)
    Output: logits [B,13]
    """
    def __init__(self, num_classes: int = 13):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),  # works if tile_size=96 -> 96/2/2/2=12
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    correct_tiles = 0
    total_tiles = 0
    for imgs, labels, adresses in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(imgs) # Output shape: [Batch, 64, 13]
        
        # Flatten outputs and labels for CrossEntropyLoss
        # Output: [Batch * 64, 13], Labels: [Batch * 64]
        loss = criterion(output.view(-1, 13), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate Tile Accuracy
        preds = output.argmax(dim=2) # [Batch, 64]
        correct_tiles += (preds == labels).sum().item()
        total_tiles += labels.numel() # Batch * 64
        
       
    return model, total_loss / len(loader), 1 - (correct_tiles / total_tiles)

def validate(model, loader, criterion, device):
    """
    Performs one epoch of validation (no gradient updates).
    """
    model.eval()
    total_loss = 0.0
    correct_tiles = 0
    total_tiles = 0
    
    with torch.no_grad():
        for imgs, labels, adresses in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            
            loss = criterion(output.view(-1, 13), labels.view(-1))
            total_loss += loss.item()
            
            preds = output.argmax(dim=2)
            correct_tiles += (preds == labels).sum().item()
            total_tiles += labels.numel()
            
    return total_loss / len(loader), 1 - (correct_tiles / total_tiles)

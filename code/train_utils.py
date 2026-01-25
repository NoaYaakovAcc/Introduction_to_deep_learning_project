import torch
from tqdm import tqdm
import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct_tiles = 0
    total_tiles = 0
    
    # Initialize Distancing Loss
    dist_criterion = DistancingLoss(margin=5.0) # Adjust margin as needed
    lambda_dist = 0.1 # Weight for the distancing loss
    
    for imgs, labels, adresses in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Unpack the tuple (logits, features)
        output, features = model(imgs) 
        
        # flatten labels for loss calculation
        flat_labels = labels.view(-1)
        flat_output = output.view(-1, 13)
        
        # 1. Classification Loss (CrossEntropy)
        loss_cls = criterion(flat_output, flat_labels)
        
        # 2. Distancing Loss (Push different classes apart)
        loss_dist = dist_criterion(features, flat_labels)
        
        # Combined Loss
        loss = loss_cls + (lambda_dist * loss_dist)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = output.argmax(dim=2) 
        correct_tiles += (preds == labels).sum().item()
        total_tiles += labels.numel()
        
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
            output, _ = model(imgs)
            
            loss = criterion(output.view(-1, 13), labels.view(-1))
            total_loss += loss.item()
            
            preds = output.argmax(dim=2)
            correct_tiles += (preds == labels).sum().item()
            total_tiles += labels.numel()
            
    return total_loss / len(loader), 1 - (correct_tiles / total_tiles)


class DistancingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Compute pairwise euclidean distances
        # embeddings: [N, d]
        n = embeddings.size(0)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Create a mask for "Negative" pairs (different labels)
        # labels: [N]
        labels = labels.view(-1, 1)
        mask_neg = (labels != labels.t()).float()
        
        # We only want to push away negatives that are too close (closer than margin)
        # Loss = ReLU(margin - distance) * Mask
        dist_loss = F.relu(self.margin - dist_matrix) * mask_neg
        
        # Average over the number of negative pairs
        return dist_loss.sum() / (mask_neg.sum() + 1e-8)

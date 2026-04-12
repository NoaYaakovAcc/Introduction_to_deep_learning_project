import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one full epoch.

    Parameters:
        model : torch.nn.Module -> The neural network to train (our model).
        loader : torch.utils.data.DataLoader -> Dataloader providing batches of - (images, labels, image_paths)
        optimizer : torch.optim.Optimizer -> Optimizer used to update model weights.
        criterion : torch.nn.Module -> Loss function used to measure prediction error
        device : torch.device -> Device on which training is performed (CPU or CUDA).

    Returns:
        tuple -> consisting of: - avgrage_loss : float -> The average batch loss across the epoch
                                - miss_rate : float -> Tile-level miss rate over the epoch
                                
                
    """
    model.train()

    total_loss = 0.0
    correct_tiles = 0
    total_tiles = 0

    # Iterate over all batches in the loader
    for images, labels, _ in loader:
        # Move data to the selected device
        images  = images.to(device)
        labels = labels.to(device)

        # Clear gradients from previous step
        optimizer.zero_grad()

        # Forward pass
        square_scores = model(images) # square_scores shape: [Batch, 64, 13]
        
        # Flatten square_scoress and labels for CrossEntropyLoss
        # square_scores: [Batch * 64, 13], Labels: [Batch * 64]
        loss = criterion(square_scores.view(-1, 13), labels.view(-1))
        
        # Backpropagation 
        loss.backward()

        # Update model parameters
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate Tile Accuracy
        preds = square_scores.argmax(dim=2) # [Batch, 64]

        correct_tiles += (preds == labels).sum().item()
        total_tiles += labels.numel() # Batch * 64
        
       
    return total_loss / len(loader), 1 - (correct_tiles / total_tiles)

def validate(model, loader, criterion, device):
    """
    Evaluate the model on one full validation epoch.

    Parameters:
        model : torch.nn.Module -> The neural network to train (our model).
        loader : torch.utils.data.DataLoader -> Dataloader providing batches of - (images, labels, image_paths)
        criterion : torch.nn.Module -> Loss function used to measure prediction error
        device : torch.device -> Device on which training is performed (CPU or CUDA).

    Returns:
    tuple -> consisting of: - avgrage_loss : float -> The average batch loss across the validation epoch
                            - miss_rate : float -> Tile-level miss rate over the validation epoch

    """
    model.eval()

    total_loss = 0.0
    correct_tiles = 0
    total_tiles = 0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            # Move current batch to the selected device
            images  = images.to(device)
            labels = labels.to(device)

            # Forward pass
            square_scores = model(images)
            
            # Compute loss over all board squares in the batch
            loss = criterion(square_scores.view(-1, 13), labels.view(-1))
            total_loss += loss.item()
            
            preds = square_scores.argmax(dim=2)

            correct_tiles += (preds == labels).sum().item()
            total_tiles += labels.numel()
            
    return total_loss / len(loader), 1 - (correct_tiles / total_tiles)

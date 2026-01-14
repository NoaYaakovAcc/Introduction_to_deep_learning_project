import torch

def evaluate_full_board_accuracy(model, loader, device):
    """
    Evaluates the model on the 'Full Board' metric.
    A prediction is considered correct ONLY if all 64 tiles on the board are classified correctly.
    """
    model.eval()
    correct_boards = 0
    total_boards = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs) # [B, 64, 13]
            preds = output.argmax(dim=2) # [B, 64]
            
            # Check per board: Are all 64 predictions equal to the 64 labels?
            # (preds == labels) returns a boolean matrix [B, 64]
            # .all(dim=1) reduces it to [B], True only if the whole row is True.
            board_correct = (preds == labels).all(dim=1)
            
            correct_boards += board_correct.sum().item()
            total_boards += imgs.size(0)
            
    acc = 100.0 * correct_boards / total_boards
    print(f"Full Board Accuracy (All 64 tiles correct): {acc:.2f}%")
    return acc
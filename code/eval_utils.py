import torch

# Dictionary to map predicted indices back to chess characters
IDX_TO_PIECE = {
    0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',   # White Pieces
    6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k', # Black Pieces
    12: '.'                                           # Empty Square
}

def indices_to_fen_string(indices):
    """
    Helper function: Converts a tensor of 64 indices (0-12) 
    back to a readable string format with '/' separators for rows.
    """
    # Move to CPU and convert to list
    idx_list = indices.cpu().numpy().tolist()
    
    # Map numbers to characters
    chars = [IDX_TO_PIECE.get(i, '?') for i in idx_list]
    
    # Group into 8 rows of 8 characters
    rows = ["".join(chars[i:i+8]) for i in range(0, 64, 8)]
    
    # Join rows with '/' to look like a real FEN
    return "/".join(rows)

def evaluate_full_board_accuracy(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    1. Calculates Strict Accuracy (all 64 tiles must be correct).
    2. Prints visual examples of Real vs. Predicted boards for debugging.
    """
    model.eval()
    
    correct_boards = 0
    total_boards = 0
    
    print("\n" + "="*60)
    print("STARTING DETAILED EVALUATION")
    print("="*60)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)  # Shape: [Batch, 64, 13]
            
            # Get predictions (argmax over the 13 classes)
            preds = torch.argmax(outputs, dim=2)  # Shape: [Batch, 64]
            
            # Check strict equality (all 64 tiles match)
            # (preds == labels) gives a boolean matrix. .all(dim=1) checks if the whole row is True.
            board_matches = (preds == labels).all(dim=1)
            
            correct_boards += board_matches.sum().item()
            total_boards += labels.size(0)
            
            # --- PRINT EXAMPLES (Only for the first batch to avoid spamming) ---
            if batch_idx == 0:
                print(f"\nPrinting first 5 examples from Batch 0:\n")
                
                # Loop over the first 5 samples in this batch (or less if batch is small)
                num_to_show = min(5, labels.size(0))
                for i in range(num_to_show):
                    true_fen = indices_to_fen_string(labels[i])
                    pred_fen = indices_to_fen_string(preds[i])
                    
                    is_correct = board_matches[i].item()
                    status = "✅ MATCH" if is_correct else "❌ FAIL"
                    
                    print(f"Sample {i} [{status}]:")
                    print(f"  TRUE: {true_fen}")
                    print(f"  PRED: {pred_fen}")
                    
                    # If failed, show exactly which indices (0-63) were wrong
                    if not is_correct:
                        diff = [j for j in range(64) if labels[i][j] != preds[i][j]]
                        print(f"  Errors at indices: {diff}")
                        
                    print("-" * 60)
            # -------------------------------------------------

    accuracy = 100.0 * correct_boards / total_boards
    
    print(f"\n" + "="*60)
    print(f"FINAL RESULT:")
    print(f"Total Boards Evaluated: {total_boards}")
    print(f"Perfectly Predicted Boards: {correct_boards}")
    print(f"Full Board Accuracy: {accuracy:.2f}%")
    print("="*60 + "\n")
    
    return accuracy
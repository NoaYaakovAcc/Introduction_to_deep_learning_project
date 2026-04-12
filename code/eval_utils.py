import matplotlib

# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg') 

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import ChessNet
from plot import plot_confusion_matrix

from torchvision import transforms
from PIL import Image



# Maps indices to characters
IDX_TO_PIECE = {
    0: 'P', 
    1: 'R', 
    2: 'N', 
    3: 'B', 
    4: 'Q', 
    5: 'K',
    6: 'p', 
    7: 'r', 
    8: 'n', 
    9: 'b', 
    10: 'q', 
    11: 'k',
    12: '.'
}

# Maps characters to unicode symbols for drawing
FEN_TO_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
    '.': ''
}

def indices_to_fen_string(indices):
    """
    Convert a 64-element tensor of predicted class indices into a board string.

    Parameters:
        indices : torch.Tensor -> Tensor of shape [64] containing predicted class indices.

    Returns:
        string -> Board string with 8 rows separated by '/' (e.g., rnbqkbnr/pppppppp/......../...etc)
        
    """
    idx_list = indices.cpu().numpy().tolist()
    chars = [IDX_TO_PIECE.get(i, '?') for i in idx_list]
    rows = ["".join(chars[i:i+8]) for i in range(0, 64, 8)]
    return "/".join(rows)

def draw_professional_board(ax, fen_str):
    """
    Draw a chessboard with pieces using Unicode chess symbols.

    Parameters:
        ax : matplotlib.axes.Axes -> Target axis on which to draw the board.
        
    fen_str : 
        string -> Board representation with 8 slash-separated rows.
    """

    light_color = "#f0d9b5"
    dark_color = "#b58863"
    
    # Draw board squares
    for y in range(8):
        for x in range(8):
            color = light_color if (x + y) % 2 == 0 else dark_color
            rect = patches.Rectangle((x, 7-y), 1, 1, linewidth=0, facecolor=color)
            ax.add_patch(rect)

    # Draw pieces    
    rows = fen_str.split('/')
    for y, row_str in enumerate(rows):
        x = 0
        for char in row_str:
            if char.isdigit():
                x += int(char)
            else:
                piece_symbol = FEN_TO_UNICODE.get(char, char)
                text_color = 'black' if char in 'pnbrqk' else 'white'
                
                # Draw subtle shadow under piece
                ax.text(
                    x + 0.5, 7.5 - y, piece_symbol, 
                    fontsize=32, ha='center', va='center', 
                    color='black', fontweight='bold', alpha=0.3
                )
                
                ax.text(
                    x + 0.5, 7.5 - y, piece_symbol, 
                    fontsize=32, ha='center', va='center', 
                    color=text_color, fontweight='normal'
                )
                x += 1
                
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.axis('off')

def clean_path_name(full_path):
    """
    Convert a full file path into a shorter, cleaner display label.

    Parameters: 
        full_path : string ->  Original file path.
       
    Returns:
        str ->  Shortened label combining game folder and filename when possible.
       
    """
    parts = full_path.split(os.sep)
    try:
        game_name = next((p for p in reversed(parts) if "game" in p), "Unknown")
        filename = parts[-1]
        return f"{game_name} | {filename}"
    except:
        return os.path.basename(full_path)

def save_visual_comparison(img_tensor, pred_fen_str, true_fen_str, clean_title, save_path, board_acc):
    """
    Saves a side-by-side visualization of:
    - the original board image
    - the predicted board rendering

    Parameters: 
        img_tensor : torch.Tensor -> Image tensor of shape [C, H, W].
        pred_fen_str : string -> Predicted board string.
        true_fen_str : string -> Ground-truth board string. Currently not drawn directly, but useful for future extension.
        clean_title : string -> Readable title for the sample.
        save_path : string -> Output path for the saved figure.
        board_acc : float -> Square-level accuracy for this board in percentage.

    Returns:
        None

    """
   
    # Denormalize image for display
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left: Original Image
    ax[0].imshow(img_np)
    ax[0].set_title(f"Original: {clean_title}", fontsize=14, fontweight='bold')
    ax[0].axis('off')
    
    # Right: Predicted Board
    draw_professional_board(ax[1], pred_fen_str)
    
    # Color coding based on accuracy
    if board_acc == 100.0:
        title_color = "green"
    elif board_acc >= 90.0:
        title_color = "#d35400" # Orange-Red
    else:
        title_color = "red"
        
    ax[1].set_title(
        f"Prediction (Acc: {board_acc:.1f}%)", 
        fontsize=14, 
        fontweight='bold', 
        color=title_color
    )

    plt.figtext(
        0.5, 
        0.05, 
        f"Pred FEN: {pred_fen_str}", 
        ha="center", 
        fontsize=10, 
        bbox={"facecolor":"white", "alpha":0.5, "pad":5}
    )
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def evaluate_full_board_accuracy(model, data_loader, device, folder_name="visual_results"):
    """
    Evaluate a trained model on a dataset and compute board-level accuracy.

    Parameters: 
        model : torch.nn.Module -> Trained model.
        data_loader : torch.utils.data.DataLoader -> Evaluation dataloader returning (images, labels, paths).
        device : torch.device ->  CPU or CUDA device used for inference.
        folder_name : str, optional -> Directory where outputs will be saved.
        

    Returns:
        float -> Perfect-board accuracy in percentage.
        
    """
    # Switch the model to evaluation mode.
    model.eval()

    correct_boards = 0
    total_boards = 0
    confusion_matrix = torch.zeros(13, 13, dtype=torch.long)

    # Create the specific folder passed from main
    os.makedirs(folder_name, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"STARTING VISUAL EVALUATION (Saving to {folder_name})")
    print("="*60)
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(data_loader):
            # Move images and labels to the selected device.
            images = images.to(device)
            labels = labels.to(device)
            
            # Run the model on the current batch.
            outputs = model(images)
            outputs = outputs.view(-1, 64, 13)
            
            preds = torch.argmax(outputs, dim=2)
            
            # Check for perfect boards
            board_matches = (preds == labels).all(dim=1)

            correct_boards += board_matches.sum().item()
            total_boards += labels.size(0)

            # Flatten predictions
            flat_preds = preds.view(-1).cpu()
            flat_labels = labels.view(-1).cpu()

            # Update confusion matrix
            for t, p in zip(flat_labels, flat_preds):
                confusion_matrix[t, p] += 1

            board_matches = (preds == labels).all(dim=1)
            correct_boards += board_matches.sum().item()
            total_boards += labels.size(0)

            if batch_idx == 0:
                num_to_save = min(10, labels.size(0))
                print(f"Saving {num_to_save} visualization images...")
                
                for i in range(num_to_save):
                    # Convert true and predicted 64-square label tensors
                    true_fen = indices_to_fen_string(labels[i])
                    pred_fen = indices_to_fen_string(preds[i])

                    clean_title = clean_path_name(paths[i])
                    
                    # Calculate specific accuracy for this board
                    correct_tiles = (preds[i] == labels[i]).sum().item()
                    board_acc = 100.0 * correct_tiles / 64.0
                    
                    safe_filename = clean_title.replace(" | ", "_").replace(".jpg", "")
                    save_name = os.path.join(folder_name, f"result_{i}_{safe_filename}.png")
                    
                    # Pass accuracy to the saving function
                    save_visual_comparison(
                        images[i], 
                        pred_fen, 
                        true_fen, 
                        clean_title, 
                        save_name, 
                        board_acc
                    )
                    
                    status = "Perfect" if board_matches[i].item() else f"NOT PERFECT {board_acc:.1f}%"
                    print(f"Sample {i} [{status}]: {clean_title}")

    # Calculate summary statistics for the confusion matrix
    total_tiles = confusion_matrix.sum().item()
    correct_all = torch.trace(confusion_matrix).item()
    overall_acc_all = 100.0 * correct_all / total_tiles if total_tiles > 0 else 0.0

    non_empty_conf = confusion_matrix[:12, :]
    non_empty_total = non_empty_conf.sum().item()
    non_empty_correct = torch.trace(non_empty_conf).item()
    overall_acc_nonempty = 100.0 * non_empty_correct / non_empty_total if non_empty_total > 0 else 0.0

    plot_confusion_matrix(
        confusion_matrix, 
        folder_name, 
        overall_acc_all, 
        overall_acc_nonempty
    )

    accuracy = 100.0 * correct_boards / total_boards
    print(f"\nFinal Perfect Board Accuracy: {accuracy:.2f}%")
    return accuracy




# REQUIRED EVALUATION FUNCTION (From Web) -> i'm not sure about this one yet

def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Mandatory evaluation function.
    Predict the board state from a single RGB image.

    Parameters:
        image : np.ndarray -> Input image with shape (H, W, 3), RGB, uint8.

    Returns:
        torch.Tensor -> A (8, 8) tensor on CPU containing class indices (int64)

    Workflow:
        1. Initialize model architecture
        2. Load saved weights
        3. Preprocess input image
        4. Run inference
        5. Return board prediction as an 8x8 tensor
    """

    # 1. Initialize model architecture
    model = ChessNet(num_classes=13)
    MODEL_PATH = 'best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load trained weights
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Ensure you have trained the model first.")

    # Move model to CPU and set to evaluation mode
    model.to(device)
    model.eval()

    # 3. Preprocessing
    transform_pipeline = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
    ])
    
    # Convert Numpy (uint8) -> PIL Image -> Tensor
    pil_img = Image.fromarray(image.astype('uint8')).convert('RGB')
    img_tensor = transform_pipeline(pil_img)
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        # Forward pass. Output shape: [1, 64, 13]
        logits = model(img_tensor)
        
        # Get class predictions (argmax). Shape: [1, 64]
        preds = torch.argmax(logits, dim=2)
        
        # Reshape to 8x8 grid as required by the spec
        board_output = preds.view(8, 8)
        
    # 5. Return requirements: strictly CPU tensor, int64 dtype
    return board_output.cpu().long()
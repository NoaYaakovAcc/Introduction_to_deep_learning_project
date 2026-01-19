import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg') 

import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
    idx_list = indices.cpu().numpy().tolist()
    chars = [IDX_TO_PIECE.get(i, '?') for i in idx_list]
    rows = ["".join(chars[i:i+8]) for i in range(0, 64, 8)]
    return "/".join(rows)

def draw_professional_board(ax, fen_str):
    light_color = "#f0d9b5"
    dark_color = "#b58863"
    
    for y in range(8):
        for x in range(8):
            color = light_color if (x + y) % 2 == 0 else dark_color
            rect = patches.Rectangle((x, 7-y), 1, 1, linewidth=0, facecolor=color)
            ax.add_patch(rect)
            
    rows = fen_str.split('/')
    for y, row_str in enumerate(rows):
        x = 0
        for char in row_str:
            if char.isdigit():
                x += int(char)
            else:
                piece_symbol = FEN_TO_UNICODE.get(char, char)
                text_color = 'black' if char in 'pnbrqk' else 'white'
                
                ax.text(x + 0.5, 7.5 - y, piece_symbol, 
                        fontsize=32, ha='center', va='center', 
                        color='black', fontweight='bold', alpha=0.3)
                
                ax.text(x + 0.5, 7.5 - y, piece_symbol, 
                        fontsize=32, ha='center', va='center', 
                        color=text_color, fontweight='normal')
                x += 1
                
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.axis('off')

def clean_path_name(full_path):
    parts = full_path.split(os.sep)
    try:
        game_name = next((p for p in reversed(parts) if "game" in p), "Unknown")
        filename = parts[-1]
        return f"{game_name} | {filename}"
    except:
        return os.path.basename(full_path)

def save_visual_comparison(img_tensor, pred_fen_str, true_fen_str, clean_title, save_path, board_acc):
    # Denormalize image for display
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
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
        
    ax[1].set_title(f"Prediction (Acc: {board_acc:.1f}%)", fontsize=14, fontweight='bold', color=title_color)
    plt.figtext(0.5, 0.05, f"Pred FEN: {pred_fen_str}", ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def evaluate_full_board_accuracy(model, data_loader, device, folder_name="visual_results"):
    model.eval()
    correct_boards = 0
    total_boards = 0
    
    # Create the specific folder passed from main
    os.makedirs(folder_name, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"STARTING VISUAL EVALUATION (Saving to {folder_name})")
    print("="*60)
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            outputs = outputs.view(-1, 64, 13)
            
            preds = torch.argmax(outputs, dim=2)
            
            # Check for perfect boards
            board_matches = (preds == labels).all(dim=1)
            correct_boards += board_matches.sum().item()
            total_boards += labels.size(0)
            
            if batch_idx == 0:
                num_to_save = min(10, labels.size(0))
                print(f"Saving {num_to_save} visualization images...")
                
                for i in range(num_to_save):
                    true_fen = indices_to_fen_string(labels[i])
                    pred_fen = indices_to_fen_string(preds[i])
                    clean_title = clean_path_name(paths[i])
                    
                    # Calculate specific accuracy for this board
                    correct_tiles = (preds[i] == labels[i]).sum().item()
                    board_acc = 100.0 * correct_tiles / 64.0
                    
                    safe_filename = clean_title.replace(" | ", "_").replace(".jpg", "")
                    save_name = os.path.join(folder_name, f"result_{i}_{safe_filename}.png")
                    
                    # Pass accuracy to the saving function
                    save_visual_comparison(images[i], pred_fen, true_fen, clean_title, save_name, board_acc)
                    
                    status = "✅" if board_matches[i].item() else f"⚠️ {board_acc:.1f}%"
                    print(f"Sample {i} [{status}]: {clean_title}")

    accuracy = 100.0 * correct_boards / total_boards
    print(f"\nFinal Perfect Board Accuracy: {accuracy:.2f}%")
    return accuracy
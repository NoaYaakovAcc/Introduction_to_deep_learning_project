import matplotlib
# This MUST happen before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

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

def plot_list(data: list[float], text1: str, text2: str, headline: str, save_dir: str = None):
    """Plots a list of doubles with custom axis labels and a title."""
    filename = headline.lower().replace(' ', '_').replace('over', '').strip('_') + '.png'
    plt.plot(data)
    plt.ylabel(text1)
    plt.xlabel(text2)
    plt.title(headline)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        plt.close()
        return full_path
    else:
        plt.close()
        return None

def plot_confusion_matrix(cm_tensor, folder_name):
    """
    Plots and saves a normalized confusion matrix (percentage table).
    X-axis: Predicted
    Y-axis: True
    """
    # 1. Normalize by Row (True Label) to get percentages
    # Add epsilon to avoid division by zero for missing classes
    row_sums = cm_tensor.sum(dim=1, keepdim=True)
    cm_perc = (cm_tensor.float() / (row_sums + 1e-9)) * 100
    
    # 2. Setup Plot
    classes = [IDX_TO_PIECE[i] for i in range(13)] # 0-12
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    im = ax.imshow(cm_perc.numpy(), interpolation='nearest', cmap='Blues')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom")
    
    # 3. Configure Axis
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    ax.set_xlabel('Predicted Label (y)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label (x)', fontsize=12, fontweight='bold')
    ax.set_title('Piece Confusion Matrix (Accuracy Rate)', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # 4. Add Text Annotations (The Percentages)
    threshold = cm_perc.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_perc[i, j].item()
            # Only show non-zero values for cleanliness, or show all
            text_color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                    color=text_color, fontsize=9)

    plt.tight_layout()
    
    # 5. Save
    save_path = os.path.join(folder_name, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Confusion Matrix saved to: {save_path}")
    
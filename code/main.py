import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from tqdm import tqdm # Import from your code

# --- Import custom modules ---
from data import scan_game, ChessBoardDataset
import plot 
from eval_utils import evaluate_full_board_accuracy
from model import ChessNet
from train_utils import train_one_epoch, validate
import data
from config import *

models = ["mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0","shufflenet_v2_x1_0","squeezenet1_0",
          "resnet18", "resnet34", "vgg16", "convnext_tiny", "densenet121", "inception_v3",  "xception",
          "resnext50_32x4d", "swin_t"]
finished_models = {"mobilenet_v2", "resnet18", "resnet34", "resnext50_32x4d", "shufflenet_v2_x1_0",
                   "xception", "inception_v3", "mobilenet_v3_small", "efficientnet_b0", "vgg16", "squeezenet1_0",
                   "convnext_tiny", "densenet121", "swin_t"}

succsessful_models = {"resnet34", "shufflenet_v2_x1_0", "vgg16", "convnext_tiny", "squeezenet1_0", "swin_t"}#models that are better than resnet18 at something

def get_all_files_in_dirs(data_root, directory_list):
    """
    Load all valueid samples from a list of game directories.

    For each directory, this function builds the expected CSV path and
    delegates the actual sample extraction to `scan_game()`.

    Parameters:
        data_root : string -> Root directory that contains all game folders.
        directory_list : list[string] -> List of game folder names to scan.

    Returns:
        list -> A combined list of all samples found across the requested folders.
    """
    all_samples = []
    for dir in directory_list:
        path = os.path.join(data_root, dir)

        # Assuming CSV filename matches game folder name prefix (e.g., game2_per_frame -> game2.csv)
        csv_path = os.path.join(path, f"{dir.split('_')[0]}.csv")

        print(f"Looking for data in: {path} | CSV: {csv_path}")
        all_samples.extend(scan_game(path, csv_path))

    return all_samples


def set_seed(seed=DEFAULT_SEED):
    """
    Set random seeds for reproducibility.

    This ensures that operations using Python random 
    start from a fixed state, which makes experiments more repeatable.

    Parameters:
        seed : int, optional -> Random seed valueue.

    Returns:
        None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_games(game_numbers):
    """
    Convert requested game numbers into folder names (e.g., [1, 2, 5] -> ['game1_per_frame', 'game2_per_frame', 'game5_per_frame'])

    Parameters:
        game_numbers : list[int] -> Numeric game identifiers.
        

    Returns:
        list[string] -> Folder names corresponding to the requested games.
        
    """
    games = []
    for game in game_numbers:
        games.append(f'game{game}_per_frame')
    return games

def main():
    """
    Main training .

    The Workflow:
        1. Defines needed configuration
        2. Loads and split data
        3. Builds datasets and loaders
        4. Initializes the model, optimizer, and loss
        5. Trains on synthetic data
        6. Evaluates zero-shot performance on real validation data
        7. Fine-tunes on real data
        8. Plot results and save final model
    """

    # 1. Configuration
    synthetic_train_samples_train_games_numbers = [2,4,5]
    real_train_samples_train_games_numbers = [6]
    eval_games_numbers = [7]
    synthetic_epochs =  DEFAULT_SYNTHETIC_EPOCHS
    real_epochs = DEFAULT_REAL_EPOCHS
    batch = DEFAULT_BATCH_SIZE
    learning_rate = DEFAULT_LEARNING_RATE

    add_noise = True
    folder_name = None 
    
    data_root = r'data'

    synthetic_train_games = get_games(synthetic_train_samples_train_games_numbers)
    real_train_games = get_games(real_train_samples_train_games_numbers)
    eval_games = get_games(eval_games_numbers)

    # Select computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    set_seed()


    # 2. Data Preparation

    # Load all samples from the requested folders
    synthetic_train_samples = get_all_files_in_dirs(data_root, synthetic_train_games)
    real_train_samples = get_all_files_in_dirs(data_root, real_train_games)
    all_eval_samples = get_all_files_in_dirs(data_root, eval_games)

    # Split by domain
    synthetic_train_samples = [s for s in synthetic_train_samples if s.domain == 'synthetic']
    real_train_samples = [s for s in real_train_samples if s.domain == 'real']
    random.shuffle(real_train_samples)

    # validation real samples split by domain
    real_eval_samples = [s for s in all_eval_samples if s.domain == 'real']

    # Output folder
    if real_epochs == 0 and folder_name is None:
        folder_name = f"ver2_visual_results_zero_shot_{MODEL_NAME}_{int(synthetic_epochs)}_epochs"
    elif folder_name is None:
        folder_name = f"ver2_visual_results_finetune_{MODEL_NAME}_{int(real_epochs)}_real epochs and {int(synthetic_epochs)}_epochs"
    
    print(f"Syntetic training set size: {len(synthetic_train_samples)}")
    print(f"Real training set size: {len(real_train_samples)}")
    print(f"validation set size: {len(real_eval_samples)}")
    
    # 3. Transforms and Loaders

    transform = transforms.Compose([
        transforms.Resize((DEFAULT_RESOLUTION, DEFAULT_RESOLUTION)),
        transforms.ToTensor(),
    ])
    
    synthetic_train_ds = ChessBoardDataset(synthetic_train_samples, transform=transform)
    real_value_ds = ChessBoardDataset(real_eval_samples, transform=transform)
    
    synthetic_train_loader = DataLoader(
        synthetic_train_ds, 
        batch_size=batch, 
        shuffle=True, 
        num_workers=DEFAULT_NUM_WORKERS
        ) # ask for a less expensive way
    
    value_loader = DataLoader(
        real_value_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=DEFAULT_NUM_WORKERS
        )
    

    # 4. Model, optimizer, and loss initialization

    # Using the custom ChessNet from model.py
    model = ChessNet(num_classes=NUMBER_OF_CHESS_CLASSES,# resolution=DEFAULT_RESOLUTION,
                      model_name=MODEL_NAME).to(device)
    
    # Stochastic Gradient Descent optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # class weights for cross-entropy loss
    class_weights = torch.ones(13)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    

    # 5. Training Loop

    print("Starting training...")

    # monitoring porpuses 
    train_losses = []
    validation_losses = []
    train_miss_rates = []
    validation_miss_rates = []
    
    max_validation_accuracy = 0.0

    # using tqdm to track progress in the terminal
    for epoch in tqdm(range(synthetic_epochs + real_epochs)):  

        # Learning-rate decay during synthetic training
        if(epoch == synthetic_epochs/3 or epoch == 2*synthetic_epochs/3):
            optimizer.param_groups[0]['lr'] /= 10

        # Switch to real-data fine-tuning phase    
        if(epoch >= synthetic_epochs):
            # Evaluate zero-shot performance before seeing real data
            if(epoch == synthetic_epochs):
                print(f"Best validation Miss Rate: {max_validation_accuracy*100:.2f}%")    

                evaluate_full_board_accuracy(
                    model,
                    value_loader, 
                    device, 
                    folder_name=(folder_name+"/zero_shot")
                    ) 
                
                # Save the zero-shot model 
                MODEL_PATH = folder_name + '/zero_shot' + '/best_model.pth'
                torch.save(model.state_dict(), MODEL_PATH)

                del synthetic_train_loader
                torch.cuda.empty_cache()

                # Build real-data training dataset and loader
                real_train_ds = ChessBoardDataset(real_train_samples, transform=transform)
                new_train_loader = DataLoader(
                    real_train_ds, 
                    batch_size=batch, 
                    shuffle=True, 
                    num_workers=DEFAULT_NUM_WORKERS
                    )
                
                optimizer.param_groups[0]['lr'] = learning_rate*10
        
        # Synthetic training with optional augmentation
        elif add_noise:
            new_train_loader = data.build_augmented_batches_loader(synthetic_train_loader)
        else:
            new_train_loader = synthetic_train_loader

        # One epoch of training
        # Using functions from train_utils.py
        train_loss, train_miss_rate  = train_one_epoch(
            model, 
            new_train_loader, 
            optimizer, 
            criterion, 
            device
            )
        
        # Validation
        validation_loss, validation_miss_rate = validate(
            model, 
            value_loader, 
            criterion, 
            device
            )
        
        # Save best-performing weights - we need to delete it ? 
        if(1 - validation_miss_rate) > max_validation_accuracy:
            best_model_wts = model.state_dict()  
            if epoch <= synthetic_epochs:
                best_zero_shot_wts = model.state_dict()
            max_validation_accuracy = 1 - validation_miss_rate

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_miss_rates.append(train_miss_rate)
        validation_miss_rates.append(validation_miss_rate)

    
    # 6. Final evalueuation
    #model.load_state_dict(best_model_wts) # Load best model weights from each train epoch

    print(f"Best validation Miss Rate: {max_validation_accuracy*100:.2f}%")
    print("Training Complete. Evaluating Full Board Accuracy...")

    if real_epochs > 0:
        folder_name = folder_name + "/fine_tune"

    # Save plots for training and validation curves    
    plot.plot_list(
        train_miss_rates, 
        "Loss", 
        "Epochs", 
        f"Training miss rate over Epochs with {real_epochs} real data epochs%", 
        save_dir=folder_name
        )
    
    plot.plot_list(
        validation_miss_rates, 
        "Loss", 
        "Epochs", 
        f"validation miss rate over Epochs with {real_epochs} real data epochs%", 
        save_dir=folder_name
        )
    
    evaluate_full_board_accuracy(
        model, 
        value_loader,
        device, 
        folder_name=folder_name
        )
    
    # 7. Save Model Weights (CRITICAL step for predict_board to work)
    MODEL_PATH = folder_name + '/best_model.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model weights saved successfully to {MODEL_PATH}")

if __name__ == '__main__':
    main()


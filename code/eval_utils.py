from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data import (
    cut_tiles_3x3, crop_board_region, id_to_piece, board_to_fen, EMPTY
)

def predict_fen_from_image(
    model,
    image_path: str,
    device: str,
    tile_size: int = 96,
    board_bbox: Optional[Tuple[int,int,int,int]] = None,
) -> str:
    model.eval()
    img = Image.open(image_path).convert("RGB")
    board_img = crop_board_region(img, board_bbox)
    tiles = cut_tiles_3x3(board_img)

    tf = transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.ToTensor(),
    ])

    batch = torch.stack([tf(t) for t in tiles], dim=0).to(device)

    with torch.no_grad():
        logits = model(batch)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    board = np.full((8, 8), EMPTY, dtype=object)
    for i, pid in enumerate(preds):
        r, c = divmod(i, 8)
        board[r, c] = id_to_piece[int(pid)]

    return board_to_fen(board)

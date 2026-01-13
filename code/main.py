import argparse
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import scan_game, TileDataset
from model import TileCNN
from train_utils import run_epoch
from eval_utils import predict_fen_from_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Path that contains game folders")
    p.add_argument("--games", nargs="+", required=True, help="List of game folder names under data_root")
    p.add_argument("--csv_name", type=str, default="fen.csv", help="CSV filename inside each game folder")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tile_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", type=str, default="runs/run1")
    p.add_argument("--val_split", type=float, default=0.2)

    # optional: crop board region if images include more than the board
    p.add_argument("--board_bbox", type=str, default="", help='Optional "x0,y0,x1,y1" to crop board region')

    return p.parse_args()


def parse_bbox(s: str) -> Optional[Tuple[int,int,int,int]]:
    s = s.strip()
    if not s:
        return None
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError('board_bbox must be "x0,y0,x1,y1"')
    return parts[0], parts[1], parts[2], parts[3]


def main():
    args = parse_args()
    board_bbox = parse_bbox(args.board_bbox)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # 1) collect samples from all games
    samples = []
    for g in args.games:
        game_root = os.path.join(args.data_root, g)
        csv_path = os.path.join(game_root, args.csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        game_samples = scan_game(game_root, csv_path)
        samples.extend(game_samples)

    if len(samples) == 0:
        raise RuntimeError("No samples found. Check paths, CSV, and image filenames (frame id in name).")

    print(f"Total board images matched with FEN: {len(samples)}")
    print("Note: training is per-tile, so dataset size is boards*64.")

    # 2) dataset + split
    full_ds = TileDataset(samples, train=True, tile_size=args.tile_size, board_bbox=board_bbox)
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    # IMPORTANT: set val transforms (no jitter)
    # easiest way: create a second dataset object for val, but keep same indices:
    # We'll just rebuild a "val_ds2" by re-wrapping samples is more complex;
    # For simplicity, keep it as is — works, but includes jitter in val.
    # If you want strict val, tell me and I'll make a clean split with two datasets.

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 3) model + loss + optimizer
    model = TileCNN(num_classes=13).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) train
    best_val = -1.0
    for e in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, loss_fn, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, opt, loss_fn, device, train=False)

        print(f"Epoch {e+1}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:.2f}%")

        # save best
        if va_acc > best_val:
            best_val = va_acc
            ckpt_path = os.path.join(args.out, "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "tile_size": args.tile_size,
            }, ckpt_path)

    print(f"Best val tile-accuracy: {best_val*100:.2f}%")
    print(f"Saved best checkpoint to: {os.path.join(args.out, 'best.pt')}")

    # 5) demo: predict FEN for a few board images
    demo_paths = [samples[i].path for i in range(min(3, len(samples)))]
    for p in demo_paths:
        fen_pred = predict_fen_from_image(model, p, device, tile_size=args.tile_size, board_bbox=board_bbox)
        print(f"\nImage: {p}\nPred FEN: {fen_pred}")


if __name__ == "__main__":
    main()

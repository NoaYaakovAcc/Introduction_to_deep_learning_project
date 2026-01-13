import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# -------------------------
# FEN <-> board utilities
# -------------------------
PIECES = "prnbqkPRNBQK"
EMPTY = "."
piece_to_id: Dict[str, int] = {EMPTY: 0}
for i, p in enumerate(PIECES, start=1):
    piece_to_id[p] = i
id_to_piece: Dict[int, str] = {v: k for k, v in piece_to_id.items()}


def fen_to_board(fen: str) -> np.ndarray:
    """Returns 8x8 array of symbols: '.', 'p','P',... based on FEN board part."""
    board_part = fen.split()[0]
    rows = board_part.split("/")
    board = []
    for r in rows:
        row = []
        for ch in r:
            if ch.isdigit():
                row.extend([EMPTY] * int(ch))
            else:
                row.append(ch)
        if len(row) != 8:
            raise ValueError(f"Bad FEN row length: {r} -> {len(row)}")
        board.append(row)
    if len(board) != 8:
        raise ValueError(f"Bad FEN rows count: {len(board)}")
    return np.array(board)


def board_to_fen(board: np.ndarray) -> str:
    """Inverse of fen_to_board for board-only FEN (no side to move, castling, etc)."""
    rows = []
    for r in board:
        s = ""
        cnt = 0
        for cell in r:
            if cell == EMPTY:
                cnt += 1
            else:
                if cnt:
                    s += str(cnt)
                    cnt = 0
                s += str(cell)
        if cnt:
            s += str(cnt)
        rows.append(s)
    return "/".join(rows)


# -------------------------
# Frame-id from filename
# -------------------------
def extract_frame_id_from_filename(path: str) -> Optional[int]:
    name = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", name)
    if not nums:
        return None
    return int(nums[-1])


def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def infer_domain_from_path(path: str) -> str:
    up = path.replace("\\", "/").upper()
    if "/TAGGED/" in up:
        return "real"
    if "/GENERATED/" in up:
        return "synthetic"
    return "unknown"


# -------------------------
# CSV rules: point/range
# -------------------------
@dataclass(frozen=True)
class FenRule:
    start: int
    end: int
    fen: str
    is_point: bool


def load_fen_rules(csv_path: str) -> List[FenRule]:
    df = pd.read_csv(csv_path)
    required = {"from_frame", "to_frame", "fen"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns {required}. Got: {set(df.columns)}")

    rules: List[FenRule] = []
    for r in df.itertuples(index=False):
        a = int(getattr(r, "from_frame"))
        b = int(getattr(r, "to_frame"))
        fen = str(getattr(r, "fen"))
        rules.append(FenRule(start=a, end=b, fen=fen, is_point=(a == b)))

    rules.sort(key=lambda x: (x.start, x.end))
    return rules


def build_point_map_and_ranges(rules: List[FenRule]) -> Tuple[Dict[int, str], List[Tuple[int, int, str]]]:
    point_map: Dict[int, str] = {}
    ranges: List[Tuple[int, int, str]] = []
    for rule in rules:
        if rule.is_point:
            point_map[rule.start] = rule.fen
        else:
            ranges.append((rule.start, rule.end, rule.fen))
    return point_map, ranges


def fen_for_frame(frame_id: int, point_map: Dict[int, str], ranges: List[Tuple[int, int, str]]) -> Optional[str]:
    if frame_id in point_map:
        return point_map[frame_id]
    for a, b, fen in ranges:
        if a <= frame_id <= b:
            return fen
    return None


# -------------------------
# Samples
# -------------------------
@dataclass
class Sample:
    path: str
    fen: str
    frame_id: int
    game: str
    domain: str


def scan_game(game_root: str, csv_path: str) -> List[Sample]:
    rules = load_fen_rules(csv_path)
    point_map, ranges = build_point_map_and_ranges(rules)

    samples: List[Sample] = []
    game_name = os.path.basename(os.path.normpath(game_root))

    for sub in ["TAGGED", "GENERATED"]:
        folder = os.path.join(game_root, sub)
        if not os.path.isdir(folder):
            continue

        for root, _, files in os.walk(folder):
            for f in files:
                p = os.path.join(root, f)
                if not is_image(p):
                    continue

                frame_id = extract_frame_id_from_filename(p)
                if frame_id is None:
                    continue

                fen = fen_for_frame(frame_id, point_map, ranges)
                if fen is None:
                    continue

                samples.append(Sample(
                    path=p,
                    fen=fen,
                    frame_id=frame_id,
                    game=game_name,
                    domain=infer_domain_from_path(p),
                ))
    return samples


# -------------------------
# Cropping board + 3x3 tiles with padding
# -------------------------
def crop_board_region(img: Image.Image, board_bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    """
    If your images are already the board only -> keep board_bbox=None.
    If not, pass board_bbox = (x0, y0, x1, y1) to crop.
    """
    if board_bbox is None:
        return img
    return img.crop(board_bbox)


def pad_board(board_img: Image.Image) -> Tuple[Image.Image, int, int]:
    w, h = board_img.size
    tw = w // 8
    th = h // 8
    padded = Image.new("RGB", (w + 2 * tw, h + 2 * th), (0, 0, 0))
    padded.paste(board_img, (tw, th))
    return padded, tw, th


def cut_tiles_3x3(board_img: Image.Image) -> List[Image.Image]:
    padded, tw, th = pad_board(board_img)
    tiles: List[Image.Image] = []
    for r in range(8):
        for c in range(8):
            x0 = c * tw
            y0 = r * th
            crop = padded.crop((x0, y0, x0 + 3 * tw, y0 + 3 * th))
            tiles.append(crop)
    return tiles


# -------------------------
# Dataset: tile classification (center piece)
# -------------------------
class TileDataset(Dataset):
    """
    Each board image becomes 64 samples:
      input: 3x3 tile crop
      label: piece in center square (13 classes)
    """
    def __init__(
        self,
        samples: List[Sample],
        train: bool = True,
        tile_size: int = 96,
        board_bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.samples = samples
        self.board_bbox = board_bbox

        # map index -> (sample_i, tile_i)
        self.index: List[Tuple[int, int]] = []
        for i in range(len(samples)):
            for t in range(64):
                self.index.append((i, t))

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((tile_size, tile_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((tile_size, tile_size)),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sample_i, tile_i = self.index[idx]
        s = self.samples[sample_i]

        img = Image.open(s.path).convert("RGB")
        board_img = crop_board_region(img, self.board_bbox)

        tiles = cut_tiles_3x3(board_img)
        tile = tiles[tile_i]

        board = fen_to_board(s.fen)
        r, c = divmod(tile_i, 8)
        piece = board[r, c]
        y = torch.tensor(piece_to_id[piece], dtype=torch.long)
        x = self.tf(tile)
        return x, y

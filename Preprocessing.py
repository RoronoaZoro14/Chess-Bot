"""
preprocess.py
=============
Run this ONCE to convert chess_games.csv → preprocessed tensors saved to disk.
After this, training never touches the CSV again.

Output files:
  data/boards.npy   shape (N, 12, 8, 8)  float32  board tensors
  data/moves.npy    shape (N,)            int32    move indices

Usage:
  python preprocess.py
  python preprocess.py --max_games 50000 --min_elo 1800 --out_dir data
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import chess
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


# ── Encoding functions ───────────────────────────────────────────────────────

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        tensor[plane, sq // 8, sq % 8] = 1.0
    return tensor


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


# ── Main preprocessing ───────────────────────────────────────────────────────

def preprocess(
    csv_path:  str = "chess_games.csv",
    out_dir:   str = "data",
    max_games: int = 50_000,
    min_elo:   int = 2000,

):
    print(f"Settings → max_games={max_games}, min_elo={min_elo}")
    os.makedirs(out_dir, exist_ok=True)
    boards_path = os.path.join(out_dir, "boards.npy")
    moves_path  = os.path.join(out_dir, "moves.npy")

    # ── Check if already done ────────────────
    if os.path.exists(boards_path) and os.path.exists(moves_path):
        boards = np.load(boards_path, mmap_mode="r")
        print(f"Preprocessed data already exists in '{out_dir}/'")
        print(f"  boards.npy : {boards.shape}  ({boards.nbytes / 1e9:.2f} GB)")
        print(f"  moves.npy  : already saved")
        print("Delete the files and re-run if you want to reprocess.")
        return

    # ── Load CSV ─────────────────────────────
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Total games in CSV: {len(df):,}")

    if "WhiteElo" in df.columns and "BlackElo" in df.columns:
        df["WhiteElo"] = pd.to_numeric(df["WhiteElo"], errors="coerce")
        df["BlackElo"] = pd.to_numeric(df["BlackElo"], errors="coerce")
        df = df.dropna(subset=["WhiteElo", "BlackElo"])
        df = df[(df["WhiteElo"] >= min_elo) & (df["BlackElo"] >= min_elo)]
        print(f"  After ELO filter (>={min_elo}): {len(df):,} games")

    df = df.head(max_games)
    print(f"  Processing up to {max_games:,} games ...")

    move_col = "AN" if "AN" in df.columns else df.columns[-1]

    # ── Parse games into numpy arrays ────────
    # Pre-allocate lists then stack — much faster than appending to arrays
    board_list = []
    move_list  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding positions"):
        pgn_text    = str(row[move_col])
        moves_clean = re.sub(r"\d+\.", "", pgn_text).split()
        board       = chess.Board()

        for san in moves_clean:
            san = san.strip()
            if not san or san in {"1-0", "0-1", "1/2-1/2", "*"}:
                break
            try:
                move = board.parse_san(san)
            except Exception:
                break

            board_list.append(board_to_tensor(board))
            move_list.append(move_to_index(move))
            board.push(move)

    # ── Stack and save ────────────────────────
    print(f"\nTotal positions encoded: {len(board_list):,}")
    print("Stacking arrays ...")

    boards_np = np.stack(board_list).astype(np.float32)  # (N, 12, 8, 8)
    moves_np  = np.array(move_list,  dtype=np.int32)     # (N,)

    print(f"Saving boards.npy  ({boards_np.nbytes / 1e9:.2f} GB) ...")
    np.save(boards_path, boards_np)

    print(f"Saving moves.npy   ({moves_np.nbytes  / 1e6:.1f} MB) ...")
    np.save(moves_path, moves_np)

    print(f"\nDone! Data saved to '{out_dir}/'")
    print(f"  boards.npy : {boards_np.shape}")
    print(f"  moves.npy  : {moves_np.shape}")
    print(f"\nNow run:  python train_from_preprocessed.py")


if __name__ == "__main__":
    preprocess()
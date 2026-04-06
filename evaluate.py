"""
evaluate.py
===========
Standalone evaluation script for the trained ChessCNN bot.
Run this any time after training to measure how good your model is.

Tests available:
  1. Top-1 and Top-5 move prediction accuracy  (uses your preprocessed data)
  2. Puzzle accuracy                            (uses Lichess puzzle CSV)

Puzzle CSV download:
  https://database.lichess.org/#puzzles
  (it's large — you can use just the first few thousand rows)

Usage:
  python evaluate.py                          # runs both tests with defaults
  python evaluate.py --test accuracy          # only move prediction accuracy
  python evaluate.py --test puzzles           # only puzzle accuracy
  python evaluate.py --test puzzles --n 1000  # test on 1000 puzzles
"""

import os
import argparse
import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Train_from_preprocessing import ChessCNN


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# Change these to match your training settings
# ═════════════════════════════════════════════════════════════════════════════

MODEL_PATH  = "chess_cnn.pt"
RES_BLOCKS  = 4
CHANNELS    = 64
DATA_DIR    = "data"
PUZZLE_CSV  = "lichess_puzzles.csv"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SHARED UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

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

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        tensor[plane, sq // 8, sq % 8] = 1.0
    return tensor


def predict_move(board: chess.Board, model: ChessCNN) -> chess.Move:
    device = next(model.parameters()).device
    tensor = torch.tensor(board_to_tensor(board),
                          dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).squeeze(0)
    legal_moves = list(board.legal_moves)
    return max(legal_moves,
               key=lambda m: logits[m.from_square * 64 + m.to_square].item())


def load_model() -> ChessCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ChessCNN(num_res_blocks=RES_BLOCKS, channels=CHANNELS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TEST 1: MOVE PREDICTION ACCURACY
#
# Loads your preprocessed boards.npy and moves.npy and measures:
#   Top-1 accuracy — did the bot predict the exact move played?
#   Top-5 accuracy — was the correct move in the bot's top 5?
# ═════════════════════════════════════════════════════════════════════════════

class PreprocessedDataset(Dataset):
    def __init__(self, boards_path, moves_path):
        self.boards = np.load(boards_path, mmap_mode="r")
        self.moves  = np.load(moves_path)

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        board = torch.tensor(np.array(self.boards[idx]), dtype=torch.float32)
        move  = torch.tensor(int(self.moves[idx]),       dtype=torch.long)
        return board, move


def test_move_accuracy(model: ChessCNN, data_dir: str = DATA_DIR,
                       batch_size: int = 256):
    """
    Measures how often the bot's predicted move matches the move
    actually played by a human in the validation dataset.
    """
    print("=" * 50)
    print("TEST 1 — Move Prediction Accuracy")
    print("=" * 50)

    boards_path = os.path.join(data_dir, "boards.npy")
    moves_path  = os.path.join(data_dir, "moves.npy")

    if not os.path.exists(boards_path) or not os.path.exists(moves_path):
        print(f"Preprocessed data not found in '{data_dir}/'")
        print("Run Preprocessing.py first.")
        return

    dataset = PreprocessedDataset(boards_path, moves_path)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    top1_correct = 0
    top5_correct = 0
    total        = 0

    with torch.no_grad():
        for boards, labels in tqdm(loader, desc="Evaluating"):
            device         = next(model.parameters()).device
            boards, labels = boards.to(device), labels.to(device)
            logits         = model(boards)

            # Top-1: highest scored move matches the label
            top1_correct += (logits.argmax(1) == labels).sum().item()

            # Top-5: correct move appears anywhere in the top 5 predictions
            top5_indices  = logits.topk(5, dim=1).indices
            top5_correct += sum(
                labels[i].item() in top5_indices[i].tolist()
                for i in range(len(labels))
            )
            total += len(labels)

    top1 = top1_correct / total * 100
    top5 = top5_correct / total * 100

    print(f"\nTotal positions tested : {total:,}")
    print(f"Top-1 accuracy         : {top1:.2f}%  (exact move match)")
    print(f"Top-5 accuracy         : {top5:.2f}%  (correct move in top 5)")
    print()
    print("What these numbers mean:")
    print(f"  Top-1 {top1:.1f}% — the bot picks the exact human move {top1:.1f}% of the time")
    print(f"  Top-5 {top5:.1f}% — the correct move is in the bot's top 5 {top5:.1f}% of the time")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TEST 2: PUZZLE ACCURACY
#
# Downloads a Lichess puzzle CSV. Each puzzle has:
#   FEN   — the board position
#   Moves — the correct move sequence (UCI format, e.g. "e2e4 d7d5")
#
# The first move in the sequence is the opponent's move (already played).
# The second move is what the bot should find.
#
# We give the bot the position AFTER the opponent's move and check
# whether it finds the correct response.
# ═════════════════════════════════════════════════════════════════════════════

def test_puzzle_accuracy(model: ChessCNN, puzzle_csv: str = PUZZLE_CSV,
                         n: int = 500):
    """
    Tests how often the bot finds the correct move in chess puzzles.
    Puzzles have one objectively correct move — this is a stricter test
    than move prediction accuracy.
    """
    print("=" * 50)
    print("TEST 2 — Puzzle Accuracy")
    print("=" * 50)

    if not os.path.exists(puzzle_csv):
        print(f"Puzzle file '{puzzle_csv}' not found.")
        print("Download from: https://database.lichess.org/#puzzles")
        print("Then place it in your Chess-Bot folder.")
        return

    df = pd.read_csv(puzzle_csv)

    # The Lichess puzzle CSV columns are:
    # PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags
    if "FEN" not in df.columns or "Moves" not in df.columns:
        print("Unexpected CSV format. Expected columns: FEN, Moves")
        print(f"Found columns: {list(df.columns)}")
        return

    df = df.head(n)
    print(f"Testing on {len(df):,} puzzles ...\n")

    correct   = 0
    skipped   = 0

    # Break down results by puzzle difficulty (rating)
    has_rating = "Rating" in df.columns
    easy_correct = easy_total = 0      # rating < 1200
    med_correct  = med_total  = 0      # 1200 <= rating < 1800
    hard_correct = hard_total = 0      # rating >= 1800

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Solving puzzles"):
        try:
            board = chess.Board(row["FEN"])
            moves = row["Moves"].split()

            if len(moves) < 2:
                skipped += 1
                continue

            # Apply the opponent's move (the puzzle setup move)
            board.push_uci(moves[0])

            # The correct answer for the bot
            correct_move = chess.Move.from_uci(moves[1])

            # Ask the bot
            predicted = predict_move(board, model)
            is_correct = (predicted == correct_move)

            if is_correct:
                correct += 1

            # Track by difficulty
            if has_rating:
                rating = int(row["Rating"])
                if rating < 1200:
                    easy_total += 1
                    if is_correct: easy_correct += 1
                elif rating < 1800:
                    med_total += 1
                    if is_correct: med_correct += 1
                else:
                    hard_total += 1
                    if is_correct: hard_correct += 1

        except Exception:
            skipped += 1
            continue

    tested = len(df) - skipped
    pct    = correct / tested * 100 if tested > 0 else 0

    print(f"\nPuzzles tested  : {tested:,}  (skipped {skipped} invalid)")
    print(f"Correct         : {correct:,}")
    print(f"Puzzle accuracy : {pct:.2f}%")

    if has_rating and tested > 0:
        print("\nBreakdown by puzzle difficulty:")
        if easy_total > 0:
            print(f"  Easy  (< 1200) : {easy_correct}/{easy_total}"
                  f"  ({easy_correct/easy_total*100:.1f}%)")
        if med_total > 0:
            print(f"  Medium (1200-1800): {med_correct}/{med_total}"
                  f"  ({med_correct/med_total*100:.1f}%)")
        if hard_total > 0:
            print(f"  Hard  (>= 1800): {hard_correct}/{hard_total}"
                  f"  ({hard_correct/hard_total*100:.1f}%)")

    print()
    print("What this means:")
    print(f"  The bot found the correct puzzle move {pct:.1f}% of the time.")
    print(f"  A random guesser from ~30 legal moves would score ~3%.")
    print(f"  A strong human player scores ~95%+ on puzzles they attempt.")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Chess CNN Bot")
    parser.add_argument("--test",       choices=["accuracy", "puzzles", "both"],
                        default="both", help="Which test to run")
    parser.add_argument("--n",          type=int, default=500,
                        help="Number of puzzles to test on")
    parser.add_argument("--data_dir",   default=DATA_DIR)
    parser.add_argument("--puzzle_csv", default=PUZZLE_CSV)
    args = parser.parse_args()

    model = load_model()

    if args.test in ("accuracy", "both"):
        test_move_accuracy(model, data_dir=args.data_dir)

    if args.test in ("puzzles", "both"):
        test_puzzle_accuracy(model, puzzle_csv=args.puzzle_csv, n=args.n)
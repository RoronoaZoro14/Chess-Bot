import os
import re
import math
import numpy as np
import pandas as pd
import chess
import chess.pgn
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# board = chess.Board()
#
# print(board)
# #df = pd.read_csv('chess_games.csv')
# #print(df.head())
#
# for sq, piece in board.piece_map().items():
#     if piece.piece_type == chess.PAWN:
#         print(sq, piece)
#
# print("Hello World")

# ──────────────────────────────────────────────
# 1.  CONSTANTS
# ──────────────────────────────────────────────

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

NUM_PLANES = 12  # piece channels
BOARD_SIZE = 8
NUM_MOVES = 4096  # 64 * 64


# ──────────────────────────────────────────────
# 2.  BOARD / MOVE ENCODING
# ──────────────────────────────────────────────

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Returns float32 array of shape (12, 8, 8).
    1.0 where the corresponding piece is present, 0.0 elsewhere.
    The board is always encoded from White's perspective.
    """
    tensor = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        row = sq // 8  # rank 0-7
        col = sq % 8  # file 0-7
        tensor[plane, row, col] = 1.0
    return tensor


def move_to_index(move: chess.Move) -> int:
    """Encode a move as from_sq * 64 + to_sq  (0..4095)."""
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int) -> chess.Move:
    """Decode a flat index back to a (possibly illegal) chess.Move."""
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)


def best_legal_move(board: chess.Board, logits: torch.Tensor) -> chess.Move:
    """
    Given raw logits over 4096 classes, return the legal move with
    the highest predicted probability.
    """
    legal_indices = [move_to_index(m) for m in board.legal_moves]
    if not legal_indices:
        return None
    legal_logits = logits[legal_indices]
    best_local = int(torch.argmax(legal_logits).item())
    best_idx = legal_indices[best_local]
    # Reconstruct the move; handle promotions (default queen)
    move = index_to_move(best_idx)
    if move not in board.legal_moves:
        # Try queen promotion
        move = chess.Move(move.from_square, move.to_square,
                          promotion=chess.QUEEN)
    return move


# ──────────────────────────────────────────────
# 3.  DATASET
# ──────────────────────────────────────────────

def parse_games(csv_path: str,
                max_games: int = 50_000,
                min_elo: int = 1500) -> list[tuple[np.ndarray, int]]:
    """
    Read chess_games.csv and extract (board_tensor, move_index) pairs.

    Expected columns (Kaggle arevel/chess-games):
        AN        – move list in Standard Algebraic Notation  e.g. '1. e4 e5 2. Nf3 ...'
        WhiteElo  – white player rating (string or int, may contain '?')
        BlackElo  – black player rating

    Returns list of (tensor, label) pairs.
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter by ELO if columns present
    if "WhiteElo" in df.columns and "BlackElo" in df.columns:
        df["WhiteElo"] = pd.to_numeric(df["WhiteElo"], errors="coerce")
        df["BlackElo"] = pd.to_numeric(df["BlackElo"], errors="coerce")
        df = df.dropna(subset=["WhiteElo", "BlackElo"])
        df = df[(df["WhiteElo"] >= min_elo) & (df["BlackElo"] >= min_elo)]
        print(f"  After ELO filter (>={min_elo}): {len(df):,} games")

    df = df.head(max_games)

    samples: list[tuple[np.ndarray, int]] = []
    move_col = "AN" if "AN" in df.columns else df.columns[-1]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing games"):
        pgn_text = str(row[move_col])
        # Strip move numbers "1. e4" → "e4"
        moves_clean = re.sub(r"\d+\.", "", pgn_text).split()
        board = chess.Board()
        for san in moves_clean:
            san = san.strip()
            if not san or san in {"1-0", "0-1", "1/2-1/2", "*"}:
                break
            try:
                move = board.parse_san(san)
            except (chess.InvalidMoveError, chess.IllegalMoveError,
                    chess.AmbiguousMoveError, ValueError):
                break
            tensor = board_to_tensor(board)
            label = move_to_index(move)
            samples.append((tensor, label))
            board.push(move)

    print(f"  Total samples: {len(samples):,}")
    return samples


class ChessDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, int]]):
        self.tensors = torch.tensor(
            np.stack([s[0] for s in samples]), dtype=torch.float32)
        self.labels = torch.tensor(
            [s[1] for s in samples], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


# ──────────────────────────────────────────────
# 4.  CNN MODEL  (ResNet-style)
# ──────────────────────────────────────────────

class ResBlock(nn.Module):
    """Basic residual block with two 3×3 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class ChessCNN(nn.Module):
    """
    Architecture
    ─────────────
    Input  : (B, 12, 8, 8)
    Stem   : Conv 3×3 → 128 channels
    Trunk  : N residual blocks
    Policy head:
        Conv 1×1 → 2 channels → flatten → FC → 4096 logits
    """

    def __init__(self, num_res_blocks: int = 10, channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )
        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_MOVES)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        x = self.policy_conv(x)
        x = x.view(x.size(0), -1)
        return self.policy_fc(x)  # raw logits (B, 4096)


# ──────────────────────────────────────────────
# 5.  TRAINING
# ──────────────────────────────────────────────

def train(
        csv_path: str = "chess_games.csv",
        save_path: str = "chess_cnn.pt",
        max_games: int = 50_000,
        min_elo: int = 1500,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 1e-3,
        val_split: float = 0.1,
        num_res_blocks: int = 10,
        channels: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────
    samples = parse_games(csv_path, max_games=max_games, min_elo=min_elo)
    dataset = ChessDataset(samples)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────
    model = ChessCNN(num_res_blocks=num_res_blocks, channels=channels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for boards, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch}/{epochs} [train]",
                                   leave=False):
            boards, labels = boards.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * boards.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += boards.size(0)

        scheduler.step()

        # ── Validate ──────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for boards, labels in val_loader:
                boards, labels = boards.to(device), labels.to(device)
                logits = model(boards)
                loss = criterion(logits, labels)
                val_loss += loss.item() * boards.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += boards.size(0)

        t_loss = train_loss / train_total
        t_acc = train_correct / train_total * 100
        v_loss = val_loss / val_total
        v_acc = val_correct / val_total * 100

        print(f"Epoch {epoch:>3} | "
              f"Train loss {t_loss:.4f}  acc {t_acc:.2f}%  | "
              f"Val loss {v_loss:.4f}  acc {v_acc:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (val acc {v_acc:.2f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    return model


# ──────────────────────────────────────────────
# 6.  INFERENCE
# ──────────────────────────────────────────────

def load_model(save_path: str = "chess_cnn.pt",
               num_res_blocks: int = 10,
               channels: int = 128) -> ChessCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessCNN(num_res_blocks=num_res_blocks, channels=channels)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_move(board: chess.Board, model: ChessCNN) -> chess.Move:
    """
    Given a chess.Board and a trained ChessCNN, return the predicted best move.
    Only legal moves are considered.
    """
    device = next(model.parameters()).device
    tensor = torch.tensor(board_to_tensor(board),
                          dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).squeeze(0)  # (4096,)
    move = best_legal_move(board, logits)
    return move


# ──────────────────────────────────────────────
# 7.  QUICK DEMO  (runs without the CSV)
# ──────────────────────────────────────────────

def demo_untrained():
    """Show encoding shapes and a random-weight prediction."""
    print("=== Demo (untrained model) ===")
    board = chess.Board()
    tensor = board_to_tensor(board)
    print(f"Board tensor shape : {tensor.shape}")  # (12, 8, 8)

    model = ChessCNN(num_res_blocks=4, channels=64)
    move = predict_move(board, model)
    print(f"Starting position  :\n{board}")
    print(f"Predicted move     : {board.san(move)}  ({move.uci()})")


# ──────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess CNN Bot")
    parser.add_argument("--mode", choices=["train", "predict", "demo"],
                        default="demo")
    parser.add_argument("--csv", default="chess_games.csv")
    parser.add_argument("--model_path", default="chess_cnn.pt")
    parser.add_argument("--max_games", type=int, default=50_000)
    parser.add_argument("--min_elo", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--res_blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--fen", type=str, default=None,
                        help="FEN string for --mode predict")
    args = parser.parse_args()

    if args.mode == "demo":
        demo_untrained()

    elif args.mode == "train":
        train(
            csv_path=args.csv,
            save_path=args.model_path,
            max_games=args.max_games,
            min_elo=args.min_elo,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_res_blocks=args.res_blocks,
            channels=args.channels,
        )

    elif args.mode == "predict":
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Train first.")
        else:
            model = load_model(args.model_path, args.res_blocks, args.channels)
            fen = args.fen or chess.STARTING_FEN
            board = chess.Board(fen)
            print(board)
            move = predict_move(board, model)
            print(f"\nPredicted best move: {board.san(move)}  ({move.uci()})")

"""
train_from_preprocessed.py
==========================
Loads preprocessed boards.npy and moves.npy and trains the CNN.
Preprocessing is completely skipped — only neural network training happens here.

Usage:
  python train_from_preprocessed.py
  python train_from_preprocessed.py --data_dir data --epochs 20 --lr 5e-4
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

#from chess_cnn_bot import ChessCNN

NUM_PLANES = 12  # piece channels
BOARD_SIZE = 8
NUM_MOVES = 4096  # 64 * 64

class ResBlock(nn.Module):
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


# ── Dataset that reads directly from .npy files ──────────────────────────────

class PreprocessedChessDataset(Dataset):
    """
    Loads boards and moves from pre-saved .npy files.
    Uses memory-mapping (mmap_mode='r') so the full array doesn't need to
    fit in RAM — numpy pages in only the slices that are actually needed.
    """
    def __init__(self, boards_path: str, moves_path: str):
        print(f"Loading {boards_path} (memory-mapped) ...")
        self.boards = np.load(boards_path, mmap_mode="r")  # (N, 12, 8, 8)
        print(f"Loading {moves_path} ...")
        self.moves  = np.load(moves_path)                  # (N,)
        print(f"  Dataset size: {len(self.moves):,} positions\n")

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        # Convert numpy slice → torch tensor on the fly (cheap, no copy needed)
        board = torch.tensor(self.boards[idx], dtype=torch.float32)
        move  = torch.tensor(int(self.moves[idx]), dtype=torch.long)
        return board, move


# ── Training loop ────────────────────────────────────────────────────────────

def train(
    data_dir:      str   = "data",
    save_path:     str   = "chess_cnn.pt",
    epochs:        int   = 10,
    batch_size:    int   = 256,
    lr:            float = 1e-3,
    val_split:     float = 0.1,
    num_res_blocks: int  = 4,
    channels:      int   = 64,
):
    boards_path = os.path.join(data_dir, "boards.npy")
    moves_path  = os.path.join(data_dir, "moves.npy")

    if not os.path.exists(boards_path) or not os.path.exists(moves_path):
        print(f"Preprocessed data not found in '{data_dir}/'")
        print("Run first:  python preprocess.py")
        return

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Device: {device}\n")

    # ── Dataset & loaders ────────────────────
    dataset = PreprocessedChessDataset(boards_path, moves_path)
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,         # 0 = no subprocess overhead (best on Windows)
        pin_memory  = use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = use_cuda,
    )

    # ── Model ────────────────────────────────
    model = ChessCNN(num_res_blocks=num_res_blocks, channels=channels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model        : {num_res_blocks} res blocks, {channels} channels")
    print(f"Parameters   : {total_params:,}")
    print(f"Train samples: {n_train:,}  |  Val samples: {n_val:,}")
    print(f"Batches/epoch: {len(train_loader):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):

        # ── Train ────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0

        for boards, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch}/{epochs} [train]",
                                   leave=False):
            boards, labels = boards.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(boards)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * boards.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += boards.size(0)

        scheduler.step()

        # ── Validate ─────────────────────────
        model.eval()
        val_loss = val_correct = val_total = 0

        with torch.no_grad():
            for boards, labels in val_loader:
                boards, labels = boards.to(device), labels.to(device)
                logits   = model(boards)
                val_loss += criterion(logits, labels).item() * boards.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += boards.size(0)

        t_acc = train_correct / train_total * 100
        v_acc = val_correct   / val_total   * 100
        t_loss_avg = train_loss / train_total
        v_loss_avg = val_loss   / val_total

        print(f"Epoch {epoch:>3} | "
              f"Train loss {t_loss_avg:.4f}  acc {t_acc:.2f}%  | "
              f"Val loss {v_loss_avg:.4f}  acc {v_acc:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (val acc {v_acc:.2f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--model_path", default="chess_cnn.pt")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--res_blocks", type=int,   default=4)
    parser.add_argument("--channels",   type=int,   default=64)
    args = parser.parse_args()

    train(
        data_dir       = args.data_dir,
        save_path      = args.model_path,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        num_res_blocks = args.res_blocks,
        channels       = args.channels,
    )
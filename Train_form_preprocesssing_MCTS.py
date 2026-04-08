"""
Train_from_preprocessing_MCTS.py
=================================
Fine-tunes the existing chess_cnn.pt using MCTS-guided self-play.

How it works:
  1. Load the existing trained model (chess_cnn.pt)
  2. Play games against itself using MCTS
  3. At each position, MCTS produces a better move distribution
     than the raw CNN alone
  4. Train the CNN on these improved distributions
  5. Save the improved model as chess_cnn_mcts.pt
  6. Repeat from step 2 with the improved model (iterative refinement)

This is the core idea behind AlphaZero — the model teaches itself
to play better by playing against itself with tree search guidance.

Output:
  chess_cnn_mcts.pt   — improved model, use this in chess_gui.py

Usage:
  python Train_from_preprocessing_MCTS.py
  python Train_from_preprocessing_MCTS.py --iterations 5 --games 50 --sims 100
"""

import os
import argparse
import random
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Train_from_preprocessing import ChessCNN
from mcts import MCTS, board_to_tensor, get_policy_and_value, cnn_value


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

INPUT_MODEL  = "chess_cnn.pt"      # model to start from
OUTPUT_MODEL = "chess_cnn_mcts.pt" # improved model saved here
RES_BLOCKS   = 4
CHANNELS     = 64


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SELF PLAY
#
# Play one full game using MCTS and collect training examples.
#
# At each position we record:
#   board_tensor  — the encoded board state  (12, 8, 8)
#   move_probs    — MCTS visit distribution over moves (4096,)
#                   This is BETTER than the raw CNN output because
#                   MCTS has searched ahead to refine the probabilities
#   outcome       — game result from this player's perspective (+1/-1/0)
#
# The CNN is then trained to match these MCTS-improved distributions.
# ═════════════════════════════════════════════════════════════════════════════

def self_play_game(model: ChessCNN,
                   simulations: int = 100,
                   max_moves:   int = 150,
                   temperature: float = 1.0
                   ) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    Play one game of chess using MCTS and return training examples.

    temperature controls move selection randomness early in the game:
      temperature=1.0  → sample proportionally to visit counts (more exploration)
      temperature=0.0  → always pick the most visited move (more exploitation)

    Returns list of (board_tensor, policy_vector, outcome) tuples.
    """
    mcts    = MCTS(model, simulations=simulations)
    board   = chess.Board()
    history = []   # store (board_tensor, policy_vector) for each position

    move_number = 0

    while not board.is_game_over() and move_number < max_moves:

        # Get MCTS move probabilities — these are the improved training labels
        move_probs = mcts.get_move_probabilities(board)

        if not move_probs:
            break

        # Encode the policy as a 4096-length vector
        policy_vector = np.zeros(4096, dtype=np.float32)
        for move, prob in move_probs.items():
            idx = move.from_square * 64 + move.to_square
            policy_vector[idx] = prob

        # Save this position and its MCTS policy
        history.append((board_to_tensor(board), policy_vector))

        # Select move — use temperature to control exploration
        # Early game: sample randomly weighted by visit counts
        # Late game:  pick the best move deterministically
        moves = list(move_probs.keys())
        probs = np.array([move_probs[m] for m in moves])

        if temperature > 0 and move_number < 30:
            # Softmax with temperature for early game variety
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
            chosen = np.random.choice(len(moves), p=probs)
        else:
            chosen = int(np.argmax(probs))

        board.push(moves[chosen])
        move_number += 1

    # ── Determine outcome ──────────────────────────────────────────────────
    # outcome is from White's perspective: +1 = White wins, -1 = Black wins
    if board.is_checkmate():
        # The side that just moved won
        outcome = 1.0 if board.turn == chess.BLACK else -1.0
    elif board.is_game_over():
        outcome = 0.0   # draw
    else:
        # Game hit max_moves — use CNN confidence as outcome estimate
        # This is consistent with how mcts.py evaluates positions,
        # avoiding the material_value pitfall of penalising sacrifices
        policy, _ = get_policy_and_value(board, model)
        outcome   = cnn_value(policy)

    # Assign outcome to each position, flipping sign based on whose turn it was
    # (positions where White moved get +outcome, Black moved get -outcome)
    examples = []
    for i, (board_tensor, policy_vector) in enumerate(history):
        # Even moves are White's turns, odd moves are Black's turns
        # (game starts with White to move)
        player_outcome = outcome if i % 2 == 0 else -outcome
        examples.append((board_tensor, policy_vector, player_outcome))

    return examples


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATASET
# ═════════════════════════════════════════════════════════════════════════════

class SelfPlayDataset(Dataset):
    """
    Dataset built from self-play games.
    Each sample is (board_tensor, policy_vector, outcome).
    """
    def __init__(self, examples: list[tuple[np.ndarray, np.ndarray, float]]):
        self.boards   = torch.tensor(
            np.stack([e[0] for e in examples]), dtype=torch.float32)
        self.policies = torch.tensor(
            np.stack([e[1] for e in examples]), dtype=torch.float32)
        self.outcomes = torch.tensor(
            [e[2] for e in examples], dtype=torch.float32)

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        return self.boards[idx], self.policies[idx], self.outcomes[idx]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING ON SELF PLAY DATA
#
# The loss has two components:
#
#   Policy loss — cross entropy between CNN output and MCTS visit distribution
#                 Trains the CNN to directly predict what MCTS found
#
#   Value loss  — MSE between CNN value estimate and actual game outcome
#                 We approximate this using the policy head's top move score
#                 (a true value head would require model architecture changes)
#
# Combined: loss = policy_loss + value_weight × value_loss
# ═════════════════════════════════════════════════════════════════════════════

def train_on_examples(model:        ChessCNN,
                      examples:     list,
                      epochs:       int   = 3,
                      batch_size:   int   = 128,
                      lr:           float = 1e-4,
                      value_weight: float = 0.5):
    """
    Fine-tune the model on self-play examples.
    Uses a smaller learning rate than initial training to avoid
    forgetting what was learned from the grandmaster games.
    """
    device  = next(model.parameters()).device
    dataset = SelfPlayDataset(examples)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=0)

    # Use a low learning rate — we're fine-tuning, not training from scratch
    optimizer = optim.Adam(model.parameters(), lr=lr)
    policy_loss_fn = nn.KLDivLoss(reduction="batchmean")  # for policy matching
    value_loss_fn  = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total      = 0

        for boards, policies, outcomes in loader:
            boards, policies, outcomes = (boards.to(device),
                                          policies.to(device),
                                          outcomes.to(device))

            optimizer.zero_grad(set_to_none=True)
            logits = model(boards)   # (B, 4096) raw logits

            # Policy loss: KL divergence between log-softmax output and MCTS policy
            log_probs   = torch.log_softmax(logits, dim=1)
            policy_loss = policy_loss_fn(log_probs, policies)

            # Value loss: use mean logit of top move as a rough value proxy
            top_logit   = logits.max(dim=1).values / 10.0  # scale to ~[-1, 1]
            value_loss  = value_loss_fn(top_logit, outcomes)

            loss = policy_loss + value_weight * value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * boards.size(0)
            total      += boards.size(0)

        print(f"    Epoch {epoch}/{epochs}  loss: {total_loss / total:.4f}")

    model.eval()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN TRAINING LOOP
#
# Iterative self-play improvement:
#   Iteration 1: load chess_cnn.pt → play 50 games → train → save
#   Iteration 2: load improved model → play 50 games → train → save
#   ...
#
# Each iteration the model gets slightly better because:
#   - It plays games using MCTS (stronger than raw CNN)
#   - It trains on MCTS move distributions (better labels than human data)
#   - The improved model then guides even better MCTS in the next iteration
# ═════════════════════════════════════════════════════════════════════════════

def load_model(path: str) -> ChessCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ChessCNN(num_res_blocks=RES_BLOCKS, channels=CHANNELS)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded {path} on {device}")
    return model


def run_mcts_training(
    input_model:  str = INPUT_MODEL,
    output_model: str = OUTPUT_MODEL,
    iterations:   int = 3,      # how many self-play → train cycles
    games:        int = 50,     # self-play games per iteration
    simulations:  int = 100,    # MCTS simulations per move
    epochs:       int = 3,      # training epochs per iteration
    batch_size:   int = 128,
    lr:           float = 1e-4,
    max_moves:    int = 150,    # max moves per game before stopping
):
    print("\n" + "═" * 55)
    print("  MCTS Self-Play Training")
    print("═" * 55)
    print(f"  Input model  : {input_model}")
    print(f"  Output model : {output_model}")
    print(f"  Iterations   : {iterations}")
    print(f"  Games/iter   : {games}")
    print(f"  Simulations  : {simulations} per move")
    print("═" * 55 + "\n")

    if not os.path.exists(input_model):
        print(f"Input model '{input_model}' not found.")
        print("Train first: python Train_from_preprocessing.py")
        return

    # Start from the existing trained model
    current_model_path = input_model
    model = load_model(current_model_path)

    for iteration in range(1, iterations + 1):
        print(f"\n{'─' * 55}")
        print(f"  Iteration {iteration}/{iterations}")
        print(f"{'─' * 55}")

        # ── Step 1: Self-play ──────────────────────────────────────────────
        print(f"\n  Playing {games} self-play games "
              f"({simulations} MCTS sims/move) ...")
        all_examples = []

        for game_num in tqdm(range(games), desc=f"  Self-play iter {iteration}"):
            examples = self_play_game(
                model,
                simulations = simulations,
                max_moves   = max_moves,
                temperature = 1.0,
            )
            all_examples.extend(examples)

        print(f"  Generated {len(all_examples):,} training positions")

        # Shuffle to break correlation between consecutive positions
        random.shuffle(all_examples)

        # ── Step 2: Train ─────────────────────────────────────────────────
        print(f"\n  Training on self-play data ({epochs} epochs) ...")
        train_on_examples(model, all_examples,
                          epochs=epochs, batch_size=batch_size, lr=lr)

        # ── Step 3: Save ──────────────────────────────────────────────────
        torch.save(model.state_dict(), output_model)
        print(f"\n  ✓ Saved improved model → {output_model}")

    print(f"\n{'═' * 55}")
    print(f"  Training complete!")
    print(f"  Final model saved to: {output_model}")
    print(f"\n  To use in chess_gui.py, change these lines:")
    print(f"    from mcts import predict_move_mcts, load_model")
    print(f"    MODEL_PATH = '{output_model}'")
    print(f"    bot_move = predict_move_mcts(board, model, simulations={simulations})")
    print(f"{'═' * 55}\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ChessCNN with MCTS self-play")
    parser.add_argument("--input_model",  default=INPUT_MODEL)
    parser.add_argument("--output_model", default=OUTPUT_MODEL)
    parser.add_argument("--iterations",   type=int,   default=3,
                        help="Number of self-play → train cycles")
    parser.add_argument("--games",        type=int,   default=50,
                        help="Self-play games per iteration")
    parser.add_argument("--sims",         type=int,   default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--epochs",       type=int,   default=3,
                        help="Training epochs per iteration")
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--max_moves",    type=int,   default=150)
    args = parser.parse_args()

    run_mcts_training(
        input_model  = args.input_model,
        output_model = args.output_model,
        iterations   = args.iterations,
        games        = args.games,
        simulations  = args.sims,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        max_moves    = args.max_moves,
    )
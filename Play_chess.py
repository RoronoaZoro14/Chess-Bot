"""
play_chess.py
=============
Interactive terminal interface to test your trained ChessCNN bot.

Modes:
  1. You (White) vs Bot (Black)
  2. Bot (White) vs You (Black)
  3. Bot vs Bot  (watch it play itself)
  4. Analyse a position  (paste a FEN, get the bot's move)

Run:
  python play_chess.py
"""

import chess
import chess.svg
import torch

# Import from your main bot file
from chess_cnn_bot import ChessCNN, load_model, predict_move, board_to_tensor

MODEL_PATH  = "chess_cnn.pt"
RES_BLOCKS  = 4
CHANNELS    = 64


# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────

def print_board(board: chess.Board, flipped: bool = False):
    """Pretty-print the board with rank/file labels."""
    ranks = range(7, -1, -1) if not flipped else range(8)
    files = "a b c d e f g h"
    print()
    for rank in ranks:
        row = f"  {rank + 1} "
        for file in range(8):
            sq    = chess.square(file, rank)
            piece = board.piece_at(sq)
            sym   = piece.unicode_symbol() if piece else "·"
            row  += sym + " "
        print(row)
    print(f"    {files}")
    print()


def print_header(text: str):
    print("\n" + "─" * 45)
    print(f"  {text}")
    print("─" * 45)


# ──────────────────────────────────────────────
# MOVE INPUT
# ──────────────────────────────────────────────

def get_human_move(board: chess.Board) -> chess.Move | None:
    """
    Ask the human for a move. Accepts:
      - SAN  e.g. 'e4', 'Nf3', 'O-O'
      - UCI  e.g. 'e2e4', 'g1f3'
      - 'quit' / 'exit'
      - 'legal' to list all legal moves
    """
    while True:
        try:
            raw = input("  Your move (SAN/UCI, 'legal', 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw.lower() in ("quit", "exit", "q"):
            return None

        if raw.lower() == "legal":
            moves = sorted(board.san(m) for m in board.legal_moves)
            print(f"  Legal moves: {', '.join(moves)}")
            continue

        # Try SAN first, then UCI
        move = None
        try:
            move = board.parse_san(raw)
        except Exception:
            pass
        if move is None:
            try:
                move = chess.Move.from_uci(raw)
                if move not in board.legal_moves:
                    move = None
            except Exception:
                pass

        if move is None:
            print(f"  ✗ Invalid or illegal move: '{raw}'. Try again.")
            continue

        return move


# ──────────────────────────────────────────────
# GAME MODES
# ──────────────────────────────────────────────

def play_game(model: ChessCNN,
              human_color: chess.Color | None = chess.WHITE):
    """
    Main game loop.
    human_color = chess.WHITE  → human plays white
    human_color = chess.BLACK  → human plays black
    human_color = None         → bot vs bot
    """
    board   = chess.Board()
    flipped = (human_color == chess.BLACK)
    mode_str = {
        chess.WHITE: "You (White) vs Bot (Black)",
        chess.BLACK: "Bot (White) vs You (Black)",
        None:        "Bot vs Bot",
    }[human_color]

    print_header(mode_str)
    print("  Type 'legal' to see all legal moves, 'quit' to exit.\n")

    move_num = 1
    while not board.is_game_over():
        print_board(board, flipped=flipped)
        turn  = board.turn
        color = "White" if turn == chess.WHITE else "Black"

        # ── Bot's turn ────────────────────────
        if human_color is None or turn != human_color:
            print(f"  [{color}] Bot is thinking...")
            move = predict_move(board, model)
            san  = board.san(move)
            print(f"  [{color}] Bot plays: {san}  ({move.uci()})")
            board.push(move)
            if human_color is None:
                input("  Press Enter for next move...")   # pause in bot-vs-bot

        # ── Human's turn ─────────────────────
        else:
            print(f"  [{color}] Your turn  (move #{move_num})")
            move = get_human_move(board)
            if move is None:
                print("\n  Game aborted.")
                return
            san = board.san(move)
            print(f"  [{color}] You played: {san}")
            board.push(move)

        if turn == chess.BLACK:
            move_num += 1

    # ── Game over ─────────────────────────────
    print_board(board)
    result = board.result()
    outcome = board.outcome()
    print_header("Game Over")
    print(f"  Result  : {result}")
    if outcome:
        if outcome.winner == chess.WHITE:
            winner = "White wins"
        elif outcome.winner == chess.BLACK:
            winner = "Black wins"
        else:
            winner = "Draw"
        print(f"  Outcome : {winner}  ({outcome.termination.name})")


# ──────────────────────────────────────────────
# ANALYSE A POSITION
# ──────────────────────────────────────────────

def analyse_position(model: ChessCNN):
    """Paste a FEN string and get the bot's top move + top-5 candidates."""
    print_header("Position Analysis")
    print("  Paste a FEN string (or press Enter for the starting position):")
    fen = input("  FEN: ").strip()
    if not fen:
        fen = chess.STARTING_FEN

    try:
        board = chess.Board(fen)
    except ValueError as e:
        print(f"  ✗ Invalid FEN: {e}")
        return

    print_board(board)

    # Get logits and show top-5 legal moves
    device = next(model.parameters()).device
    tensor = torch.tensor(board_to_tensor(board),
                          dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).squeeze(0)

    # Score all legal moves
    legal_moves  = list(board.legal_moves)
    move_scores  = []
    for m in legal_moves:
        idx   = m.from_square * 64 + m.to_square
        score = logits[idx].item()
        move_scores.append((score, m))

    move_scores.sort(reverse=True)

    print(f"  Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"  Legal moves available: {len(legal_moves)}")
    print("\n  Top-5 candidate moves (by model score):")
    print(f"  {'Rank':<6} {'Move':<8} {'UCI':<8} {'Score':>8}")
    print("  " + "─" * 34)
    for i, (score, move) in enumerate(move_scores[:5], 1):
        san = board.san(move)
        print(f"  {i:<6} {san:<8} {move.uci():<8} {score:>8.3f}")

    best = move_scores[0][1]
    print(f"\n  ✓ Best move: {board.san(best)}  ({best.uci()})")


# ──────────────────────────────────────────────
# MENU
# ──────────────────────────────────────────────

def main():
    print("\n" + "═" * 45)
    print("        ♟  Chess CNN Bot  ♟")
    print("═" * 45)

    # Load model
    try:
        model = load_model(MODEL_PATH, RES_BLOCKS, CHANNELS)
        print(f"\n  ✓ Model loaded from '{MODEL_PATH}'")
    except FileNotFoundError:
        print(f"\n  ✗ Model file '{MODEL_PATH}' not found.")
        print("    Train first:  python chess_cnn_bot.py --mode train --csv chess_games.csv")
        return

    while True:
        print("\n  Select mode:")
        print("    1 · You (White) vs Bot (Black)")
        print("    2 · Bot (White) vs You (Black)")
        print("    3 · Bot vs Bot")
        print("    4 · Analyse a position (FEN)")
        print("    5 · Quit")

        choice = input("\n  Choice: ").strip()

        if choice == "1":
            play_game(model, human_color=chess.WHITE)
        elif choice == "2":
            play_game(model, human_color=chess.BLACK)
        elif choice == "3":
            play_game(model, human_color=None)
        elif choice == "4":
            analyse_position(model)
        elif choice == "5":
            print("\n  Goodbye! ♟\n")
            break
        else:
            print("  Invalid choice. Enter 1–5.")


if __name__ == "__main__":
    main()
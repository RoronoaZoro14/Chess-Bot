"""
chess_gui.py
============
Pygame chess GUI with an initial menu screen.
Play against the trained ChessCNN bot, watch it play itself, or play as Black.

Place this file in the same folder as:
  - Train_from_preprocessing.py  (contains ChessCNN, ResBlock)
  - chess_cnn.pt                 (your trained model weights)

Controls:
  - Click a piece to select it (highlighted in yellow)
  - Click a destination square to move
  - Press R to return to the menu
  - Press Q or close the window to quit
"""

import sys
import time
import chess
import torch
import numpy as np
import pygame

from Train_from_preprocessing import ChessCNN


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "chess_cnn.pt"
RES_BLOCKS = 4
CHANNELS   = 64

WINDOW_SIZE = 640
SQ_SIZE     = WINDOW_SIZE // 8

# Board colors
COLOR_LIGHT  = (240, 217, 181)
COLOR_DARK   = (181, 136,  99)
COLOR_SELECT = (247, 247, 105)
COLOR_LEGAL  = ( 50, 205,  50)
COLOR_BLACK  = (  0,   0,   0)
COLOR_WHITE  = (255, 255, 255)
COLOR_STATUS = ( 40,  40,  40)

# Menu colors
COLOR_BG         = ( 30,  30,  30)
COLOR_TITLE      = (240, 200, 100)
COLOR_BTN        = ( 70,  70,  70)
COLOR_BTN_HOVER  = (100, 100, 100)
COLOR_BTN_TEXT   = (255, 255, 255)
COLOR_BTN_BORDER = (200, 200, 200)

PIECE_UNICODE = {
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.KING,   chess.WHITE): "♔",
    (chess.PAWN,   chess.BLACK): "♟",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.KING,   chess.BLACK): "♚",
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COORDINATE CONVERSION
# ═════════════════════════════════════════════════════════════════════════════

def square_to_pixels(square: int) -> tuple[int, int]:
    col = chess.square_file(square)
    row = chess.square_rank(square)
    x   = col * SQ_SIZE
    y   = (7 - row) * SQ_SIZE
    return (x, y)


def pixels_to_square(x: int, y: int) -> int:
    col = x // SQ_SIZE
    row = 7 - (y // SQ_SIZE)
    if 0 <= col <= 7 and 0 <= row <= 7:
        return chess.square(col, row)
    return -1


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL LOADING & INFERENCE
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


def load_model() -> ChessCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ChessCNN(num_res_blocks=RES_BLOCKS, channels=CHANNELS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def predict_move(board: chess.Board, model: ChessCNN) -> chess.Move:
    device = next(model.parameters()).device
    tensor = torch.tensor(board_to_tensor(board),
                          dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).squeeze(0)
    legal_moves = list(board.legal_moves)
    best_move   = max(legal_moves,
                      key=lambda m: logits[m.from_square * 64 + m.to_square].item())
    return best_move


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DRAWING THE BOARD
# ═════════════════════════════════════════════════════════════════════════════

def draw_board(surface, board, selected_sq, legal_targets):

    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            x, y   = square_to_pixels(square)
            if square == selected_sq:
                color = COLOR_SELECT
            elif (rank + file) % 2 == 0:
                color = COLOR_DARK
            else:
                color = COLOR_LIGHT
            pygame.draw.rect(surface, color, (x, y, SQ_SIZE, SQ_SIZE))

    for target_sq in legal_targets:
        x, y   = square_to_pixels(target_sq)
        cx, cy = x + SQ_SIZE // 2, y + SQ_SIZE // 2
        pygame.draw.circle(surface, COLOR_LEGAL, (cx, cy), SQ_SIZE // 8)

    font = pygame.font.SysFont("segoeuisymbol", int(SQ_SIZE * 0.8))
    for square, piece in board.piece_map().items():
        symbol = PIECE_UNICODE[(piece.piece_type, piece.color)]
        x, y   = square_to_pixels(square)
        shadow = font.render(symbol, True, COLOR_BLACK)
        surface.blit(shadow, (x + 3, y + 3))
        piece_color = COLOR_WHITE if piece.color == chess.WHITE else COLOR_BLACK
        text = font.render(symbol, True, piece_color)
        surface.blit(text, (x + 1, y + 1))


def draw_status(surface, board, message, window_h):
    pygame.draw.rect(surface, COLOR_STATUS, (0, window_h, WINDOW_SIZE, 40))
    font = pygame.font.SysFont("arial", 18)
    if board.is_game_over():
        result = board.result()
        term   = board.outcome().termination.name if board.outcome() else ""
        status = f"Game Over — {result}  ({term})   R = menu"
    else:
        turn   = "White" if board.turn == chess.WHITE else "Black"
        check  = "  ⚠ CHECK!" if board.is_check() else ""
        status = f"{message}  |  {turn}'s turn{check}   R = menu"
    surface.blit(font.render(status, True, COLOR_WHITE), (10, window_h + 10))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MENU
#
# draw_menu()  renders the menu every frame.
# show_menu()  runs the event loop and returns the chosen human_color:
#                chess.WHITE → human plays White
#                chess.BLACK → human plays Black
#                None        → Bot vs Bot
# ═════════════════════════════════════════════════════════════════════════════

def draw_menu(surface, buttons, mouse_pos):
    surface.fill(COLOR_BG)

    # Icon + title
    icon_font  = pygame.font.SysFont("segoeuisymbol", 60)
    title_font = pygame.font.SysFont("arial", 52, bold=True)
    sub_font   = pygame.font.SysFont("arial", 22)

    icon = icon_font.render("♟", True, COLOR_TITLE)
    surface.blit(icon, icon.get_rect(center=(WINDOW_SIZE // 2, 100)))

    title = title_font.render("Chess CNN Bot", True, COLOR_TITLE)
    surface.blit(title, title.get_rect(center=(WINDOW_SIZE // 2, 170)))

    sub = sub_font.render("Select a game mode to start", True, (180, 180, 180))
    surface.blit(sub, sub.get_rect(center=(WINDOW_SIZE // 2, 215)))

    # Buttons — hover effect via collidepoint()
    btn_font = pygame.font.SysFont("arial", 24, bold=True)
    for btn in buttons:
        rect     = btn["rect"]
        hovered  = rect.collidepoint(mouse_pos)
        bg       = COLOR_BTN_HOVER if hovered else COLOR_BTN
        pygame.draw.rect(surface, bg,              rect, border_radius=8)
        pygame.draw.rect(surface, COLOR_BTN_BORDER, rect, width=2, border_radius=8)
        label = btn_font.render(btn["label"], True, COLOR_BTN_TEXT)
        surface.blit(label, label.get_rect(center=rect.center))

    # Footer
    foot = pygame.font.SysFont("arial", 16).render(
        "Q / Esc to quit", True, (120, 120, 120))
    surface.blit(foot, foot.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE + 20)))


def show_menu(surface):
    """Blocks until the player picks a mode. Returns human_color."""

    btn_w, btn_h = 340, 55
    btn_x = (WINDOW_SIZE - btn_w) // 2

    buttons = [
        {"rect": pygame.Rect(btn_x, 280, btn_w, btn_h),
         "label": "▶  You (White) vs Bot (Black)",
         "value": chess.WHITE},
        {"rect": pygame.Rect(btn_x, 355, btn_w, btn_h),
         "label": "▶  Bot (White) vs You (Black)",
         "value": chess.BLACK},
        {"rect": pygame.Rect(btn_x, 430, btn_w, btn_h),
         "label": "▶  Bot vs Bot",
         "value": None},
    ]

    clock = pygame.time.Clock()
    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn["rect"].collidepoint(event.pos):
                        return btn["value"]

        draw_menu(surface, buttons, mouse_pos)
        pygame.display.flip()
        clock.tick(60)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN GAME LOOP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    window_h = WINDOW_SIZE + 40
    screen   = pygame.display.set_mode((WINDOW_SIZE, window_h))
    pygame.display.set_caption("Chess CNN Bot")
    clock    = pygame.time.Clock()

    try:
        model = load_model()
    except FileNotFoundError:
        print(f"Model '{MODEL_PATH}' not found. Train first.")
        pygame.quit(); sys.exit()

    # Outer loop: menu → game → menu → game ...
    while True:

        # ── Phase 1: Menu ─────────────────────────────────────────────────────
        human_color = show_menu(screen)

        # ── Phase 2: Game ─────────────────────────────────────────────────────
        board         = chess.Board()
        selected_sq   = None
        legal_targets = []
        if human_color == chess.WHITE:
            message = "Your turn"
        elif human_color == chess.BLACK:
            message = "Bot is thinking..."
        else:
            message = "Press Enter to make the bots move"       # Used in Bot vs. Bot matches

        # If human plays Black, bot makes the first move immediately
        if human_color == chess.BLACK:
            bot_move = predict_move(board, model)
            board.push(bot_move)
            message = f"Bot played {bot_move.uci()} — your turn"

        running = True
        bot_move_ready = False

        while running:

            # ── Events ───────────────────────────────────────────────────────
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_r:
                        running = False          # back to menu
                    elif event.key == pygame.K_RETURN and human_color is None:
                        bot_move_ready = True

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if (human_color is not None
                            and not board.is_game_over()
                            and board.turn == human_color):

                        mx, my     = event.pos
                        clicked_sq = pixels_to_square(mx, my)

                        if clicked_sq == -1:
                            selected_sq, legal_targets = None, []

                        elif selected_sq is None:
                            # First click — select piece
                            piece = board.piece_at(clicked_sq)
                            if piece and piece.color == human_color:
                                selected_sq   = clicked_sq
                                legal_targets = [m.to_square for m in board.legal_moves
                                                 if m.from_square == clicked_sq]
                                message = f"Selected {piece.symbol().upper()} — pick destination"
                            else:
                                message = "Select one of your pieces first"

                        else:
                            # Second click — execute move
                            if clicked_sq in legal_targets:
                                move  = chess.Move(selected_sq, clicked_sq)
                                piece = board.piece_at(selected_sq)
                                if (piece and piece.piece_type == chess.PAWN
                                        and chess.square_rank(clicked_sq) in (0, 7)):
                                    move = chess.Move(selected_sq, clicked_sq,
                                                      promotion=chess.QUEEN)
                                board.push(move)
                                selected_sq, legal_targets = None, []
                                message = "Bot is thinking..."

                                # Redraw before bot thinks so status updates immediately
                                screen.fill(COLOR_DARK)
                                draw_board(screen, board, None, [])
                                draw_status(screen, board, message, WINDOW_SIZE)
                                pygame.display.flip()

                                if not board.is_game_over():
                                    bot_move = predict_move(board, model)
                                    board.push(bot_move)
                                    message = f"Bot played {bot_move.uci()} — your turn"

                            elif (board.piece_at(clicked_sq) and
                                  board.piece_at(clicked_sq).color == human_color):
                                selected_sq   = clicked_sq
                                legal_targets = [m.to_square for m in board.legal_moves
                                                 if m.from_square == clicked_sq]
                                message = "Piece re-selected"
                            else:
                                selected_sq, legal_targets = None, []
                                message = "Invalid — re-select a piece"

            # ── Bot vs Bot auto-move ──────────────────────────────────────────
            if (human_color is None and not board.is_game_over() and running and bot_move_ready):
                bot_move = predict_move(board, model)
                san      = board.san(bot_move)
                board.push(bot_move)
                color   = "White" if not board.turn else "Black"
                message = f"{color} played {san} - press Enter for next move"
                bot_move_ready = False          #resetting the flag, wait for next Enter

            # ── Draw ─────────────────────────────────────────────────────────
            screen.fill(COLOR_DARK)
            draw_board(screen, board, selected_sq, legal_targets)
            draw_status(screen, board, message, WINDOW_SIZE)
            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    main()
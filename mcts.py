"""
mcts.py
=======
Monte Carlo Tree Search engine for the ChessCNN bot.

This file is standalone — import predict_move_mcts() from here
and use it as a drop-in replacement for the original predict_move().

Usage in chess_gui.py:
    # Replace:
    from Train_from_preprocessing import ChessCNN
    # With:
    from mcts import predict_move_mcts, load_model

    # Replace:
    bot_move = predict_move(board, model)
    # With:
    bot_move = predict_move_mcts(board, model, simulations=200)
"""

import math
import chess
import numpy as np
import torch

from Train_from_preprocessing import ChessCNN


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

MODEL_PATH  = "chess_cnn.pt"
RES_BLOCKS  = 4
CHANNELS    = 64

# MCTS hyperparameters
C_PUCT       = 1.4    # exploration constant in UCB formula
#                       higher = explore more, lower = exploit more
SIMULATIONS  = 200    # number of MCTS iterations per move
#                       more = stronger but slower


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BOARD ENCODING (same as other files)
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CNN INFERENCE HELPERS
#
# get_policy_and_value() is the core function MCTS calls on every new node.
# It returns:
#   policy — a probability distribution over all 4096 moves
#   value  — a score in [-1, 1] estimating how good the position is
#             for the side to move (positive = good, negative = bad)
#
# WHY CNN-BASED VALUE:
#   The old material_value() heuristic was actively harmful for puzzles
#   because puzzles often require sacrificing material to win. The material
#   counter would score sacrifices as -1.0 (terrible) even when they lead
#   to checkmate, causing MCTS to avoid the correct move entirely.
#
#   Instead we use the CNN's own confidence as the value estimate:
#     value = max(policy probabilities) scaled to [-1, 1]
#   This means "if the CNN is very confident about one move, the position
#   is probably good (high value). If all moves look equally bad, low value."
#   This aligns with what the CNN actually learned from grandmaster games.
# ═════════════════════════════════════════════════════════════════════════════

def cnn_value(policy: dict) -> float:
    """
    Derive a position value from the CNN policy distribution.

    Uses the maximum policy probability scaled to [-1, 1]:
      value = max_prob * 2 - 1

    Intuition:
      - If CNN is very confident (max_prob near 1.0) → value near +1.0
        (the position is clear, one move dominates)
      - If CNN is uncertain (max_prob near 0.0) → value near -1.0
        (the position is unclear or bad — no good move stands out)

    This respects what the CNN learned — including that sacrifices
    leading to checkmate are GOOD moves, not material losses.
    """
    if not policy:
        return 0.0
    return float(max(policy.values())) * 2.0 - 1.0


def terminal_value(board: chess.Board) -> float:
    """
    Returns exact value for terminal (game over) positions.
    These are the only positions where we use a rule-based value
    because the outcome is certain and known.
    """
    if board.is_checkmate():
        return -1.0   # side to move is in checkmate — they lost
    return 0.0        # stalemate, draw by repetition, etc.


def get_policy_and_value(board: chess.Board,
                          model: ChessCNN) -> tuple[dict, float]:
    """
    Runs the CNN on the current board and returns:
      policy : dict mapping chess.Move → prior probability
      value  : float in [-1, 1] estimating position quality for side to move

    Only legal moves are included in the policy dict.
    Their probabilities are re-normalised to sum to 1.
    Value is derived from the CNN policy (not material count).
    """
    device = next(model.parameters()).device
    tensor = torch.tensor(board_to_tensor(board),
                          dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor).squeeze(0)   # shape (4096,)

    # Convert logits → probabilities with softmax
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    # Extract probabilities for legal moves only
    legal_moves = list(board.legal_moves)
    policy = {}
    total  = 0.0
    for move in legal_moves:
        idx          = move.from_square * 64 + move.to_square
        policy[move] = float(probs[idx])
        total       += float(probs[idx])

    # Re-normalise so legal move probs sum to 1
    if total > 0:
        policy = {m: p / total for m, p in policy.items()}
    else:
        policy = {m: 1.0 / len(legal_moves) for m in legal_moves}

    # Value derived from CNN confidence — NOT material count
    value = cnn_value(policy)

    return policy, value


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MCTS NODE
#
# Each node in the search tree represents a board position.
# It stores:
#   board         — the chess position at this node
#   parent        — the node we came from
#   move          — the move that led here from the parent
#   children      — dict of move → child MCTSNode
#   prior         — CNN prior probability P(s, a) for this node's move
#   visit_count   — N(s, a): how many times this node was visited
#   value_sum     — sum of all values backpropagated through this node
#                   Q(s, a) = value_sum / visit_count
# ═════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    def __init__(self, board: chess.Board,
                 parent: "MCTSNode | None" = None,
                 move:   chess.Move | None = None,
                 prior:  float = 0.0):
        self.board       = board.copy()
        self.parent      = parent
        self.move        = move
        self.prior       = prior

        self.children    : dict[chess.Move, MCTSNode] = {}
        self.visit_count : int   = 0
        self.value_sum   : float = 0.0

        # Unexplored legal moves — we expand one at a time
        self.untried_moves = list(board.legal_moves)

    @property
    def q_value(self) -> float:
        """Average value of this node (exploitation term)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_fully_expanded(self) -> bool:
        """True when all legal moves have been tried at least once."""
        return len(self.untried_moves) == 0

    @property
    def is_terminal(self) -> bool:
        """True when the game is over at this node."""
        return self.board.is_game_over()

    def ucb_score(self, c_puct: float = C_PUCT) -> float:
        """
        UCB score used to select which child to visit next.

        UCB = Q(s,a) + C × P(s,a) × √N(parent) / (1 + N(s,a))

        Q(s,a)  — exploitation: how good has this move been?
        P(s,a)  — prior: how much does the CNN like this move?
        N terms — exploration: prefer less-visited nodes
        """
        if self.parent is None:
            return 0.0
        parent_visits = self.parent.visit_count
        exploration   = (c_puct * self.prior *
                         math.sqrt(parent_visits) / (1 + self.visit_count))
        return self.q_value + exploration

    def best_child(self, c_puct: float = C_PUCT) -> "MCTSNode":
        """Return the child with the highest UCB score."""
        return max(self.children.values(),
                   key=lambda child: child.ucb_score(c_puct))

    def expand(self, move: chess.Move, prior: float) -> "MCTSNode":
        """
        Create a new child node by making a move.
        Removes the move from untried_moves and adds it to children.
        """
        child_board = self.board.copy()
        child_board.push(move)
        child = MCTSNode(child_board, parent=self, move=move, prior=prior)
        self.untried_moves.remove(move)
        self.children[move] = child
        return child

    def backpropagate(self, value: float):
        """
        Walk up the tree updating visit counts and value sums.
        The value is negated at each level because the perspective
        alternates between players (what's good for me is bad for you).
        """
        self.visit_count += 1
        self.value_sum   += value
        if self.parent:
            self.parent.backpropagate(-value)   # flip sign for opponent


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MCTS SEARCH
#
# The main search loop. For each simulation:
#   1. Selection   — walk down the tree using UCB scores
#   2. Expansion   — add one new child node
#   3. Evaluation  — ask the CNN for policy + value at the new node
#   4. Backprop    — send the value back up the tree
#
# After all simulations, return the move whose child was visited most.
# Most visits = most confidence = best move.
# ═════════════════════════════════════════════════════════════════════════════

class MCTS:
    def __init__(self, model: ChessCNN,
                 simulations: int   = SIMULATIONS,
                 c_puct:      float = C_PUCT):
        self.model       = model
        self.simulations = simulations
        self.c_puct      = c_puct

    def search(self, board: chess.Board) -> chess.Move:
        """
        Run MCTS from the given board position and return the best move.
        """
        root = MCTSNode(board)

        # Initialise root with CNN policy so expansion uses prior probabilities
        policy, _ = get_policy_and_value(board, self.model)

        for simulation in range(self.simulations):

            # ── Step 1: Selection ─────────────────────────────────────────────
            # Walk down the tree picking the best child by UCB until we reach
            # a node that still has untried moves, or a terminal node.
            node = root
            while node.is_fully_expanded and not node.is_terminal:
                node = node.best_child(self.c_puct)

            # ── Step 2: Expansion ─────────────────────────────────────────────
            # If the node is not terminal and has untried moves, expand one.
            if not node.is_terminal and node.untried_moves:
                # Use CNN policy to pick which untried move to expand next
                # (prefer moves the CNN thinks are promising)
                move_priors = {
                    m: policy.get(m, 1.0 / len(node.untried_moves))
                    for m in node.untried_moves
                }
                move  = max(move_priors, key=move_priors.get)
                prior = move_priors[move]
                node  = node.expand(move, prior)

                # Get fresh policy and value for the new node
                if not node.is_terminal:
                    policy, value = get_policy_and_value(node.board, self.model)
                else:
                    value = terminal_value(node.board)
            else:
                # Terminal node — use exact game result as value
                value = terminal_value(node.board)

            # ── Step 3 & 4: Backpropagation ───────────────────────────────────
            node.backpropagate(value)

        # After all simulations, pick the move visited most often
        if not root.children:
            # Fallback if no simulations completed
            return next(iter(board.legal_moves))

        best_move = max(root.children,
                        key=lambda m: root.children[m].visit_count)
        return best_move

    def get_move_probabilities(self, board: chess.Board) -> dict[chess.Move, float]:
        """
        Run search and return visit count distribution over moves.
        Used during self-play training to generate better training labels.
        """
        root = MCTSNode(board)
        policy, _ = get_policy_and_value(board, self.model)

        for _ in range(self.simulations):
            node = root
            while node.is_fully_expanded and not node.is_terminal:
                node = node.best_child(self.c_puct)
            if not node.is_terminal and node.untried_moves:
                move_priors = {
                    m: policy.get(m, 1.0 / len(node.untried_moves))
                    for m in node.untried_moves
                }
                move  = max(move_priors, key=move_priors.get)
                node  = node.expand(move, move_priors[move])
                if not node.is_terminal:
                    policy, value = get_policy_and_value(node.board, self.model)
                else:
                    value = terminal_value(node.board)
            else:
                value = terminal_value(node.board)
            node.backpropagate(value)

        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits == 0:
            return {}
        return {
            move: child.visit_count / total_visits
            for move, child in root.children.items()
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PUBLIC API
# These are the functions you call from chess_gui.py and evaluate.py
# ═════════════════════════════════════════════════════════════════════════════

def load_model(model_path:  str = MODEL_PATH,
               res_blocks:  int = RES_BLOCKS,
               channels:    int = CHANNELS) -> ChessCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ChessCNN(num_res_blocks=res_blocks, channels=channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def predict_move_mcts(board:       chess.Board,
                      model:       ChessCNN,
                      simulations: int   = SIMULATIONS,
                      c_puct:      float = C_PUCT) -> chess.Move:
    """
    Drop-in replacement for predict_move() that uses MCTS.
    Takes longer but plays significantly stronger.

    simulations=50   → fast, moderate improvement
    simulations=200  → slower, strong improvement
    simulations=800  → slow, very strong (use on GPU)
    """
    mcts = MCTS(model, simulations=simulations, c_puct=c_puct)
    return mcts.search(board)
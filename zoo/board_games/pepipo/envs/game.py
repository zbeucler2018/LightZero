from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

# TODO: turn into string enum
class Colors:
  BLACK = "\033[30m"
  RED = "\033[31m"
  GREEN = "\033[32m"
  YELLOW = "\033[33m"
  BLUE = "\033[34m"
  MAGENTA = "\033[35m"
  CYAN = "\033[36m"
  WHITE = "\033[37m"
  UNDERLINE = "\033[4m"
  RESET = "\033[0m"


PLAYER_COLOR_MAP = {
   "player_0": Colors.GREEN,   # (0, 255, 0)
   "player_1": Colors.BLUE,    # (0, 0, 255)
   "player_2": Colors.RED,     # (255, 0, 0)
   "player_3": Colors.MAGENTA  # (255, 0, 255)
}


class t_Piece(Enum):
  EMPTY = 0
  PI = 1
  PE = 2
  PO = 3


@dataclass
class Piece:
  _typename: t_Piece
  player_id: Optional[str|float] = ""
  color: Optional[Colors] = ""

  def to_str(self) -> str:
    match self._typename:
      case t_Piece.EMPTY:
        return self.color + " " + Colors.RESET
      case t_Piece.PI:
        return self.color + "." + Colors.RESET
      case t_Piece.PE:
        return self.color + "O" + Colors.RESET
      case t_Piece.PO:
        return self.color + "0" + Colors.RESET
      case _:
        return self.color + "?" + Colors.RESET


class Board:
  def __init__(self):
    self.max_pieces_per_spot_on_board = 2
    self.board_size = 8
    self.board = []
    self.empty_board()

  def empty_board(self) -> None:
    self.board = [ [Piece(t_Piece.EMPTY) for _ in range(self.max_pieces_per_spot_on_board)] for _ in range(self.board_size**self.max_pieces_per_spot_on_board) ]

  def convert_xy_to_indx(self, x: int, y: int) -> int:
    """Converts (x,y) coordinates to an index on the board"""
    return x + y * self.board_size

  def __setitem__(self, key, value):
    x, y = key
    indx = self.convert_xy_to_indx(x, y)
    if isinstance(value, list):
      self.board[indx] = value
    elif isinstance(value, Piece):
      # pi's go on the left and everything else
      # on the right for rendering purposes
      if value._typename == t_Piece.PI:
        self.board[indx][0] = value
      else:
        self.board[indx][1] = value

  def __getitem__(self, key):
    x, y = key
    index = self.convert_xy_to_indx(x, y)
    return self.board[index]

  def __repr__(self) -> str:
    """Renders the board to the console"""
    tmp = "   |0 ||1 ||2 ||3 ||4 ||5 ||6 || 7|\n\n"
    for y in reversed(range(self.board_size)):  # Iterate in reverse for correct orientation
        tmp += f"{y}| "
        for x in range(self.board_size):
          indx = self.convert_xy_to_indx(x, y)
          l, r = self.board[indx]
          tmp += f"|{l.to_str()}{r.to_str()}|"
        tmp += "\n"
    tmp += "\n"
    return tmp


class Game:

  def __init__(self, n_players: int = 2, log_level: int = logging.INFO):
    self.n_players = n_players
    self.board = Board()
    self.max_pos_per_player = 8
    self.po_per_player = {"player_0": self.max_pos_per_player, "player_1": self.max_pos_per_player}
    self.n_pieces_in_a_row_to_win = 5 # Need to get 5 in a row to win

    # setup the logger
    self.logger = logging.getLogger("Game")
    self.logger.setLevel(log_level)
    self.logger.addHandler(logging.StreamHandler())


    assert (self.n_players < 5) and (self.n_players > 1), f"Invalid amount of players. PePiPo is played with 2-4 players, not {self.n_players}"
    if self.n_players > 2: raise NotImplementedError(f"2 players are only supported at the moment, not {self.n_players}")

  def play(self) -> None:
    raise NotImplementedError()

  def print_board(self) -> None:
    print(self.board)

  def validate_move(self, x: int, y: int, piece_type: t_Piece, player_id: str) -> bool:
    """Returns true if the move is valid.
    1. PE's and PO's can only be placed in empty spaces.
    2. PI's can only be placed within PE's.
    3. Player has not used all 8 of their PO's.
    4. Move is within the board.
    """
    reason = "encountered move that doesn't meet any of the rules"
    is_valid = False
    # 1. PEs and POs can only be placed in empty spaces
    if piece_type == t_Piece.PE and self.board[x, y][1]._typename == t_Piece.EMPTY:
       reason = "PE is placed in an empty spot"
       is_valid = True
    if (piece_type == t_Piece.PO) and self.board[x, y][1]._typename == t_Piece.EMPTY:
      reason = "PO is placed in an empty spot"
      is_valid = True
    # 2. PIs can only be placed within empty PEs
    if piece_type == t_Piece.PI and self.board[x, y][1]._typename == t_Piece.PE and self.board[x,y][0]._typename == t_Piece.EMPTY:
      reason = "PI is placed in an empty PE"
      is_valid = True
    # 3. Player has used all 8 of their PO's
    if piece_type == t_Piece.PO and self.po_per_player[player_id] == 0:
      reason = "no more PO's left"
      is_valid = False
    # 4. Move is within the board
    if x < 0 or x >= self.board.board_size or y < 0 or y >= self.board.board_size:
      reason = "outside of board"
      is_valid = False
    self.logger.debug(f"validate_move() | is_valid: {is_valid} reason: {reason} ({x}, {y}) {piece_type.name} {player_id}")
    return is_valid

  def make_move(self, x: int, y: int, piece_type: t_Piece, player_id: str) -> None:
    """Places a piece on the board."""
    piece = Piece(piece_type, player_id=player_id, color=PLAYER_COLOR_MAP[player_id])
    self.board[x, y] = piece
    if piece._typename == t_Piece.PO: # decrement player PO count
        self.po_per_player[player_id] = self.po_per_player[player_id] - 1

  def check_tie(self, player_id: str) -> bool:
    """Returns True if there is no more valid moves, False if not.
    There is a tie game if there are no more valid moves left on the board.
    """
    for x in range(self.board.board_size):
        for y in range(self.board.board_size):
            for p in (t_Piece.PE, t_Piece.PI, t_Piece.PO):
                if self.validate_move(x, y, p, player_id):
                    return False
    return True

  def check_winner(self, player_id: str) -> bool:
    """Returns True if the current player has won, False if not.
    To win, the player must have 5 pieces in a row diagonally, horizontally, or vertically.
    Also, a space with a PE and a PI count for both players.
    """
    def check_current_player_in_space(space: list[Piece, Piece]):
        return space[0].player_id == player_id or space[1].player_id == player_id

    def check_diagonal(start_x: int, start_y: int, dx: int, dy: int) -> bool:
        """Checks a diagonal starting from (start_x, start_y) in direction (dx, dy)."""
        diagonal_pieces = [self.board[start_x + i*dx, start_y + i*dy] for i in range(self.n_pieces_in_a_row_to_win)]
        return all(check_current_player_in_space(dp) for dp in diagonal_pieces)

    # Check horizontal
    for y in range(self.board.board_size):
        for x in range(self.board.board_size - self.n_pieces_in_a_row_to_win + 1):
            board_spaces = [self.board[x+w, y] for w in range(self.n_pieces_in_a_row_to_win)]
            if all(check_current_player_in_space(bs) for bs in board_spaces):
                self.logger.debug(f"{player_id} won horizontally")
                return True

    # Check vertical
    for x in range(self.board.board_size):
        for y in range(self.board.board_size - self.n_pieces_in_a_row_to_win + 1):
            board_spaces = [self.board[x, y+w] for w in range(self.n_pieces_in_a_row_to_win)]
            if all(check_current_player_in_space(bs) for bs in board_spaces):
                self.logger.debug(f"{player_id} won vertically")
                return True

    # Check diagonals (top-left to bottom-right)
    for y in range(self.board.board_size - self.n_pieces_in_a_row_to_win + 1):
        for x in range(self.board.board_size - self.n_pieces_in_a_row_to_win + 1):
            if check_diagonal(x, y, 1, 1):
                self.logger.debug(f"{player_id} won top-left-bottom-right diagonally")
                return True

    # Check diagonals (top-right to bottom-left)
    for y in range(self.n_pieces_in_a_row_to_win - 1, self.board.board_size):
        for x in range(self.board.board_size - self.n_pieces_in_a_row_to_win + 1):
            if check_diagonal(x, y, 1, -1):
                self.logger.debug(f"{player_id} won top-right-bottom-left diagonally")
                return True
    return False

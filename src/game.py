import torch

class HexGame():
  def __init__(self, k: int) -> None:
    self.k = k
    self.board = torch.tensor([[0 for _ in range(k)] for _ in range(k)], dtype=torch.float32)
    zeros = torch.tensor([[False for _ in range(k)] for _ in range(k)])
    self.occupied_slots = torch.logical_or(zeros, self.board)
    self.occupied_slots.requires_grad = False
    self.ones = torch.ones((k, k))
    self.player_1_layer = torch.zeros((k, k))
    self.player_2_layer = torch.zeros((k, k))

  def prettyprint(self):
    lines = []    
    for i in range(self.k):
      line = []
      for j in range(i+1):
        line.append(self.board[i-j,j].item())
      lines.append(line)
    for i in range(self.k-1):
      line = []
      for j in range((self.k-1)):
        try:
          elem = self.board[(self.k-1) - j + i, j+1]
          elem = elem.item()
          line.append(elem)
        except:
          continue    
      lines.append(line)
    for line in lines:
      linestr = " " * (self.k - len(line))
      for elem in line:
        linestr = linestr + str(elem)
        if elem > 9:
          linestr += ' '
        else:
          linestr += '  '
      print(linestr)

    
    
  def from_state(self, board: torch.Tensor):
    self.board = board.clone().detach()
    zeros = torch.tensor([[False for _ in range(len(board))] for _ in range(len(board))])
    self.occupied_slots = torch.logical_or(zeros, self.board)
    self.occupied_slots.requires_grad = False
    self.player_1_layer = torch.where(self.board == 1, 1, 0)
    self.player_2_layer = torch.where(self.board == 2, 1, 0)


  def get_state(self):
    return self.board.clone().detach()

  def move(self, i: int, j: int, player: int):
    self.board[i, j] = player
    self.occupied_slots[i, j] = True
    if player == 1:
      self.player_1_layer[i, j] = 1
    if player == 2:
      self.player_2_layer[i, j] = 1




  #def get_model_input(self, current_player):
  #  """ 
  #  Model takes 4 channels as input. First channel is all 1s if player one is moving, 0 otherwise. 
  #  Second channel is all 1s if player two is moving, 0 otherwise. Third channel contains player 1's moves.
  #  Fourth channel contains player 2's moves.
  #  """
  #  modelinput = torch.zeros((1, 2, self.k, self.k))
  #  if current_player == 1:
  #    modelinput[0, 0, :, :] = self.player_1_layer
  #    modelinput[0, 1, :, :] = self.player_2_layer
  #  if current_player == 2:
  #    modelinput[0, 0, :, :] = self.player_2_layer.T
  #    modelinput[0, 1, :, :] = self.player_1_layer.T
  #  return modelinput
  
  def get_model_input(self, current_player):
    """ 
    Model takes 4 channels as input. First channel is all 1s if player one is moving, 0 otherwise. 
    Second channel is all 1s if player two is moving, 0 otherwise. Third channel contains player 1's moves.
    Fourth channel contains player 2's moves.
    """
    modelinput = torch.zeros((1, 2, self.k, self.k))
    modelinput[0, 0, :, :] = self.board
    if current_player == 1:
      modelinput[0, 1, :, :] = 0
    if current_player == 2:
      modelinput[0, 1, :, :] = 1
    return modelinput


  # Consider moving to rust? https://hackernoon.com/calling-rust-from-python-with-pyo3
  # Use numba?
  def check_has_player_won(self, player_zero_one: 0 | 1):
    start_position_exists = False
    end_position_exists = False
    dfs_startpoints = []

    # Check and set start/end positions if they exist.
    for i in range(self.k):
      if(player_zero_one == 0 and self.board[i, 0] == 1):
        start_position_exists = True
        dfs_startpoints.append(torch.tensor([i, 0]))
      if(player_zero_one == 1 and self.board[0, i] == 2):
        start_position_exists = True
        dfs_startpoints.append(torch.tensor([0, i]))
      if(player_zero_one == 0 and self.board[i, self.k-1] == 1):
        end_position_exists = True
      if(player_zero_one == 1 and self.board[self.k-1, i] == 2):
        end_position_exists = True
    
    # Start or end position has not been claimed, return false
    if not (start_position_exists or end_position_exists):
      return False
    
    # Otherwise, run dfs to check if path exists
    board_boolean = torch.where(self.board == (player_zero_one + 1), False, True)
    for starting_vertex in dfs_startpoints:
      closed = board_boolean.clone().detach()
      closed[starting_vertex[0], starting_vertex[1]] = True
      path_exists_from_starting_vertex = self.dfs(starting_vertex, closed, player_zero_one)
      if path_exists_from_starting_vertex:
        return True
    return False


  def dfs(self, current, closed, player_zero_one: 0 | 1):
    if player_zero_one == 0 and current[1] == self.k-1:
      return True
    if player_zero_one == 1 and current[0] == self.k-1:
      return True
    
    closed[current[0], current[1]] = True
    # TODO: Move to helper function
    nbors = torch.tensor([[current[0] - 1, current[1]], 
      [current[0] - 1, current[1]+1],
      [current[0], current[1] - 1],
      [current[0], current[1] + 1],
      [current[0]+1, current[1] - 1],
      [current[0]+1, current[1]],
    ])
    
    for nbor in nbors:
      # If neighbor is out of bounds, ignore it.
      if nbor[0] < 0 or nbor[0] >= self.k or nbor[1] < 0 or nbor[1] >= self.k:
        continue

      # If neighbor has been visited, ignore it
      if closed[nbor[0], nbor[1]]:
        continue

      # Otherwise, recurse on neighbor
      in_subpath = self.dfs(nbor, closed, player_zero_one)
      if in_subpath:
        return in_subpath
    return False


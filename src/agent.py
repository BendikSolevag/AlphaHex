import torch
from game import HexGame
from tree import MCT, Node


class Agent:
  def __init__(self, tree, search_iterations):
    self.tree: MCT = tree
    self.search_iterations: int = search_iterations

  def move(self):
    self.tree.search(self.search_iterations)
    return self.tree.select_action()
  
  def move_model_only(self, game: HexGame, noise: torch.Tensor):
    """ For use when we are not consulting MCTS to perform moves. """
    modelinput = game.get_model_input(self.tree.root.player)
    move_probabilities: torch.Tensor = self.tree.model.forward(modelinput)[0]
    move_probabilities = move_probabilities + noise
    mask = torch.logical_not(torch.logical_or(game.board, torch.zeros((game.k, game.k))))
    if self.tree.root.player == 2:
      mask = mask.T
    mask = mask.flatten()
    move_probabilities = move_probabilities * mask
    move = torch.argmax(move_probabilities)
    move = (int(move / game.k), (move % game.k).item())
    if self.tree.root.player == 2:
      move = (move[1], move[0])
    return move

  def cleanup(self, move, updated_state, current_player):
    # If move has been expanded as result of search, select existing child
    if move in self.tree.root.children:
      self.tree.root = self.tree.root.children[move]
      self.tree.root.parent = None
    # Otherwise, create new root. Should only happen in 1st iteration (before player 2 has made init search, or in MODEL mode)
    else:
      self.tree.root = Node((current_player % 2)+ 1, None, updated_state)
      self.tree.root.board_state[move[0], move[1]] = current_player


def switch_agent(current_agent, other_agent):
  tmp = current_agent
  current_agent = other_agent
  other_agent = tmp
  return current_agent, other_agent

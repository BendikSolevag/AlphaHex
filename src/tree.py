import random
import torch
from game import HexGame
import time

# Each search state should include which player is moving from it
class MCT:
  def __init__(self, current_player, board_state, model, critic, hex_board_size, explore_coefficient):
    self.root = Node(current_player, None, board_state)
    self.model = model
    self.critic = critic
    self.hex_board_size = hex_board_size
    self.explore_coefficient = explore_coefficient


  def select_action(self):
    """ 
    Selects action based on which of the root's child vertices has the best value sum for the current player
    """
    # Move based on which child vertex was visited most times.
    highest_sum = None
    highest_sum_action = None

    for action, child in self.root.children.items():
      if highest_sum is None:
        highest_sum = child.value_sum
        highest_sum_action = action
        continue

      if self.root.player == 1 and child.value_sum  >= highest_sum:
        highest_sum = child.value_sum
        highest_sum_action = action

      if self.root.player == 2 and child.value_sum  <= highest_sum:
        highest_sum = (child.value_sum / child.visit_count)
        highest_sum_action = action
    
    
    if highest_sum_action is None:
      raise ValueError(f'Attempting to play from a vertex with no children: \n{self.root.board_state}')
    if not self.root.board_state[highest_sum_action[0], highest_sum_action[1]] == 0:
      raise ValueError('Attempting to play in occupied spot')

    return highest_sum_action
  

  def search(self, search_iterations: int):
    """
    param: search_iterations (int) - Takes number of search iterations per move as input. 
    If 0 is given, the tree will search for 0.9 seconds before returning.

    For each search iteration, follow the tree policy from the tree root to a leaf vertex.
    Populate the leaf vertex's children as all possible successor board states.
    Perform rollout for leaf vertex, and update path from leaf to root.
    """

    search_start = time.time()
    n_searches = 0
    
    while True:
      # If search_iterations is set, break after set number of iterations
      if (search_iterations > 0) and n_searches > search_iterations:
        break
      # Otherwise, allow model to search for 0.9 seconds.
      if (search_iterations == 0) and (time.time() - search_start >= 0.9):
        break
      
      # Update function state variables
      n_searches += 1
      current_vertex = self.root

      # Resolve path down to leaf vertex
      while len(current_vertex.children):
        top_policy_value = None
        top_actions = []

        for action, child in current_vertex.children.items():
          Q_value = child.value_sum / child.visit_count
          explore_factor = self.explore_coefficient * torch.sqrt(torch.log(current_vertex.visit_count) / (1 + child.visit_count))
          policy_value = Q_value  

          if current_vertex.player == 1:
            policy_value += explore_factor
        
            if top_policy_value is None or policy_value >= top_policy_value:
              top_actions.append(action)
              top_policy_value = policy_value
              if policy_value > top_policy_value:
                top_actions = [action]

          if current_vertex.player == 2:
            policy_value -= explore_factor
            if top_policy_value is None or policy_value <= top_policy_value:
              top_actions.append(action)
              top_policy_value = policy_value
              if policy_value < top_policy_value:
                top_actions = [action]
        
        selected_child = top_actions[torch.randint(0, len(top_actions), (1,))]
        current_vertex = current_vertex.children[selected_child]

      # Expand identified leaf vertex by populating children
      mask = torch.logical_not(torch.logical_or(torch.flatten(current_vertex.board_state.clone().detach()), torch.zeros((self.hex_board_size**2))))
      for i in range(mask.shape[0]):
        if not mask[i]:
          continue
        action = (int(i / self.hex_board_size), (i % self.hex_board_size))
        current_vertex.children[action] = Node(
          (current_vertex.player % 2) + 1, 
          current_vertex, 
          current_vertex.board_state.clone().detach()
        )
        current_vertex.children[action].board_state[action[0], action[1]] = current_vertex.player


      # Perform rollout for expanded vertex.
      z = self.perform_rollout(current_vertex)

      # Backpropagate path, update value sums and scores. Set current vertex back to root.
      while current_vertex.parent is not None:
        current_vertex.value_sum += z
        current_vertex.visit_count += 1
        current_vertex = current_vertex.parent
      current_vertex.value_sum += z
      current_vertex.visit_count += 1
    

  def perform_rollout(self, current_vertex):
    # Initialise neccesary state variables
    rollout_player = current_vertex.player
    rollout_number = 0
    z = -1
    mcgamestate = HexGame(self.hex_board_size)
    mcgamestate.from_state(current_vertex.board_state)

    # 50% of the time, use value network to approximate value of leaf vertex
    if random.random() > 0.5:
      modelinput = mcgamestate.get_model_input(current_vertex.player)
      approx =  self.critic(modelinput)
      return approx.item()
    
    # The rest of the time, use the actor network to play until the game terminates.
    rollout_discount = 0.99
    while True:
      # Check if game has terminated
      if mcgamestate.check_has_player_won(rollout_player - 1):
        if rollout_player == 1:
          z = 1
        break
      if rollout_number >= self.hex_board_size**2 - 1:
        if current_vertex.player == 1:
          z = -0.5
        if current_vertex.player == 2:
          z = 0.5
        break
      rollout_number += 1

      # If not, move for the current player and parse the output
      modelinput = mcgamestate.get_model_input(rollout_player)
      move_probabilities: torch.Tensor = self.model.forward(modelinput)[0]
      mask = torch.logical_not(torch.logical_or(mcgamestate.board, torch.zeros((self.hex_board_size, self.hex_board_size))))
      mask = mask.flatten()
      move_probabilities = move_probabilities * mask  
      move = torch.argmax(move_probabilities)
      (i, j) = (int(move / self.hex_board_size), (move % self.hex_board_size).item())
      
      # Make the move
      if not len(current_vertex.children) == 0:
        current_vertex = current_vertex.children[(i, j)]
      mcgamestate.move(i, j, rollout_player)

      # Get ready for next round by switching player and decreasing rollout discount
      if rollout_player == 1:
        rollout_player = 2
      elif rollout_player == 2:
        rollout_player = 1
      rollout_discount = rollout_discount*0.99
    
    return z * rollout_discount
  

class Node:
  def __init__(self, player, parent, board_state):
    self.player: int = player
    self.parent: Node | None = parent
    self.board_state: torch.Tensor = board_state
    self.children: dict[tuple[int, int], Node] = {}
    self.visit_count: torch.Tensor = torch.tensor(1)
    self.value_sum: torch.Tensor = torch.tensor(0, dtype=torch.float)

  
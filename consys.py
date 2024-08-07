import torch
from agent import Agent, switch_agent
from game import HexGame
from model import ANET, CNET
from tree import MCT

def switch_player(current_player):
  if(current_player == 1):
    return 2
  elif(current_player == 2):
    return 1

def episode(model1: ANET, model2: ANET, critic1: CNET, critic2: CNET, hex_board_size: int, search_games_per_move: int, explore_coefficient: float, starting_player: int, mode: str):
  """ 
  Runs a single iteration of the Hex game with the supplied models and hyperparameters.
  
  param: model1 - (Our) actor network
  param: model2 - (Opponent) actor network
  param: critic1 - (Our) value network
  param: critic2 - (Opponent) value network
  param: hex_board_size - height/width of the hex board
  param: search_games_per_move - Number of search iterations to run at each tree search. If 0 is given, the tree will search for 0.9 seconds.
  param: explore_coefficient - Explore coefficient applied to the tree policy
  starting_player: - Whether player 1 or player 2 starts the game
  """
  game = HexGame(hex_board_size)
  current_player = starting_player
  tree_one = MCT(current_player, game.get_state(), model1, critic1, hex_board_size, explore_coefficient)
  tree_two = MCT(current_player, game.get_state(), model2, critic2, hex_board_size, explore_coefficient)

  current_agent = Agent(tree_one, search_games_per_move)
  other_agent = Agent(tree_two, search_games_per_move)  
  if current_player == 2:
    current_agent = Agent(tree_two, search_games_per_move)
    other_agent = Agent(tree_one, search_games_per_move)
    

  move_id = 0
  movelist = []
  while True:
    move = None
    if mode == 'MCTS':
      move = current_agent.move()
    elif mode == 'MODEL':
      noise = torch.zeros((game.k**2))
      if current_player == 2:
        noise = torch.rand((game.k**2)) * explore_coefficient
      move = current_agent.move_model_only(game, noise)
 
    # Add move+state to state history
    modelinput = game.get_model_input(current_player)
    movelist.append((modelinput, move, current_player))
    
    game.move(move[0], move[1], current_player)


    # Prune both agents' trees
    current_agent.cleanup(move, game.get_state(), current_player)
    other_agent.cleanup(move, game.get_state(), current_player)
    
    # Update state variables, check if game has terminated.
    current_player = switch_player(current_player)
    current_agent, other_agent = switch_agent(current_agent, other_agent)
    
    # If the game terminates, return
    if move_id >= hex_board_size**2:
      # The game was a draw. Set reward to -0.5
      return -0.5, movelist, game
    
    # The game was won from our perspective. Set reward 1.
    if current_player == 2 and game.check_has_player_won(0):
      return 1, movelist, game
    
    # The game was lost from our perspective. Set reward -1.
    if current_player == 1 and game.check_has_player_won(1):
      return -1, movelist, game
    
    # Iterate move id
    move_id += 1

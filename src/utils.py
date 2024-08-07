import os
import torch
from datetime import datetime
from model import ANET, CNET

def create_model_save_directory():
  now = datetime.now()
  model_save_path = f"./models/{str(now)}-models"
  os.mkdir(model_save_path)
  os.mkdir(f'{model_save_path}/actor')
  os.mkdir(f'{model_save_path}/critic')
  return model_save_path


def save_model(model, epoch, path, model_initialising_parameters, prefix):
  torch.save(model.state_dict(), f'{path}/{prefix}/net-{str(epoch)}.pt')
  torch.save(model_initialising_parameters, f'{path}/model-initialising-parameters.pt')


def parse_movelist(reward, movelist, hex_board_size):
    current_moves_list = []
    for modelinput, move, player in movelist:

      label = None
      if reward == -0.5:
        label = torch.zeros((1, hex_board_size**2), dtype=torch.float)
        label[0, move([0]*hex_board_size) + move[1]] = 1

      if reward == -1:
        # If the game was a loss, record player two's moves as winning moves
        if player == 2:
          label = torch.zeros((1, hex_board_size**2), dtype=torch.float)
          reversed_move = (move[1], move[0])
          label[0, (reversed_move[0] * hex_board_size) + reversed_move[1]] = 1
        # ... and record our moves as losing moves.
        if player == 1:
          label = torch.ones((1, hex_board_size**2), dtype=torch.float)
          label[0, (move[0] * hex_board_size) + move[1]] = 0
          
      if reward == 1:
        # If the game was a win, record player one's moves as winning moves
        if player == 1:
          label = torch.zeros((1, hex_board_size**2), dtype=torch.float)
          label[0, (move[0] * hex_board_size) + move[1]] = 1
        # ... and record player two's moves as losing moves
        if player == 2:
          label = torch.ones((1, hex_board_size**2), dtype=torch.float)
          reversed_move = (move[1], move[0])
          label[0, (reversed_move[0] * hex_board_size) + reversed_move[1]] = 0
          
      if label is None:
        raise ValueError('Unexpected error, label was not set during game states resolution.')
      current_moves_list.append((modelinput, label))
    return current_moves_list


def parse_cnet_movelist(reward, movelist):
    cnet_moves_list = []
    for modelinput, move, player in movelist:
      label = None
      if reward == -0.5:
        label = torch.tensor([-0.5], dtype=torch.float)

      if reward == -1:
        # If the game was a loss, record player two's moves as winning moves
        if player == 2:
          label = torch.tensor([1], dtype=torch.float)
        # ... and record our moves as losing moves.
        if player == 1:
          label = torch.tensor([-1], dtype=torch.float)
          
      if reward == 1:
        # If the game was a win, record player one's moves as winning moves
        if player == 1:
          label = torch.tensor([1], dtype=torch.float)
        # ... and record player two's moves as losing moves
        if player == 2:
          label = torch.tensor([-1], dtype=torch.float)

      if label is None:
        raise ValueError('Unexpected error, label was not set during game states resolution.')
      cnet_moves_list.append((modelinput, label))
    return cnet_moves_list


def initialise_models(model_initialising_parameters):
  model_actor = ANET(
    model_initialising_parameters["n_layers"], 
    model_initialising_parameters["input_channels"], 
    model_initialising_parameters["kernel_size"], 
    model_initialising_parameters["activation_function"], 
    model_initialising_parameters["skip_connection"]
  )
  model_critic = CNET(
    model_initialising_parameters["n_layers"], 
    model_initialising_parameters["input_channels"], 
    model_initialising_parameters["kernel_size"], 
    model_initialising_parameters["activation_function"], 
    model_initialising_parameters["skip_connection"]
  )
  opponent_actor = ANET(
    model_initialising_parameters["n_layers"], 
    model_initialising_parameters["input_channels"], 
    model_initialising_parameters["kernel_size"], 
    model_initialising_parameters["activation_function"], 
    model_initialising_parameters["skip_connection"]
  )
  opponent_critic = CNET(
    model_initialising_parameters["n_layers"], 
    model_initialising_parameters["input_channels"], 
    model_initialising_parameters["kernel_size"], 
    model_initialising_parameters["activation_function"], 
    model_initialising_parameters["skip_connection"]
  )
  return model_actor, model_critic, opponent_actor, opponent_critic

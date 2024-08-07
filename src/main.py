import os
import random
import torch
from tqdm import tqdm

from config import config
from consys import episode
from model import backpropagate_anet, backpropagate_cnet, resolve_lossfunction, resolve_optimizer
from utils import create_model_save_directory, initialise_models, parse_cnet_movelist, parse_movelist, save_model

def train(
  hex_board_size,
  search_games_per_move,
  learning_rate,
  weight_decay,
  n_layers,
  kernel_size,
  activation_function,
  skip_connection,
  batch_size,
  name_optimizer,
  name_lossfunction,
  n_models_to_save,
  bootstrap_samples,
  episodes,
):
  input_channels = 2
  model_initialising_parameters = {
    "n_layers": n_layers,
    "input_channels": input_channels,
    "kernel_size": kernel_size,
    "activation_function": activation_function,
    "skip_connection": skip_connection
  }

  # Initialise neccesary state variables
  model_actor, model_critic, opponent_actor, opponent_critic = initialise_models(model_initialising_parameters)
  optimizer_actor = resolve_optimizer(model_actor.parameters(), name_optimizer, learning_rate, weight_decay)
  optimizer_critic = resolve_optimizer(model_critic.parameters(), name_optimizer, learning_rate, weight_decay)
  lossfunc = resolve_lossfunction(name_lossfunction)
  model_save_path = create_model_save_directory()
  previous_models = [(model_actor.state_dict(), model_critic.state_dict())]
  global_movelist = []
  global_cnet_movelist = []
  save_model(model_actor, 0, model_save_path, model_initialising_parameters, 'actor')
  save_model(model_critic, 0, model_save_path, model_initialising_parameters, 'critic')
  pbar = tqdm(total=(n_models_to_save * episodes), desc="Training")

  # Start training loop
  for m_savespot in range(n_models_to_save):
    # Make previous model available as opponent for training.
    previous_models.append((model_actor.state_dict(), model_critic.state_dict()))
    for epoch in range(episodes):
      pbar.update(1)
      # Select random opponent to train against.
      next_opponent_actor, next_opponent_critic = random.choice(previous_models)
      opponent_actor.load_state_dict(next_opponent_actor)
      opponent_critic.load_state_dict(next_opponent_critic)

      # Run episode
      reward, movelist, game = episode(model_actor, opponent_actor, model_critic, opponent_critic, hex_board_size, search_games_per_move, 1, (epoch % 2) + 1, 'MCTS')

      # Parse gamestates to trainable dataset
      global_movelist = global_movelist + parse_movelist(reward, movelist, game.k)
      global_cnet_movelist = global_cnet_movelist + parse_cnet_movelist(reward, movelist)

      # Update model
      backpropagate_anet(model_actor, optimizer_actor, lossfunc, global_movelist, batch_size, bootstrap_samples)
      backpropagate_cnet(model_critic, optimizer_critic, lossfunc, global_cnet_movelist, batch_size, bootstrap_samples)

    # Save models to partake in TOPP
    save_model(model_actor, m_savespot+1, model_save_path, model_initialising_parameters, 'actor')
    save_model(model_critic, m_savespot+1, model_save_path, model_initialising_parameters, 'critic')
    
  return model_save_path


def topp(
hex_board_size,
search_games_per_move,
round_robin_rematches,
explore_coefficient,
models_path
):
  # Retrieve saved models
  model_initialising_parameters = torch.load(f'{models_path}/model-initialising-parameters.pt')
  model_actor, model_critic, opponent_actor, opponent_critic = initialise_models(model_initialising_parameters)
  models_names = os.listdir(f'{models_path}/actor')
  models_names = [model for model in models_names if model.startswith('net')]
  models_names.sort()

  for model_path in models_names:
    for opponent_path in models_names:
      if model_path == opponent_path:
        continue
      # Load trained models from save location
      model_actor.load_state_dict(torch.load(f'{models_path}/actor/{model_path}'))
      model_critic.load_state_dict(torch.load(f'{models_path}/critic/{model_path}'))      
      opponent_actor.load_state_dict(torch.load(f'{models_path}/actor/{opponent_path}'))
      opponent_critic.load_state_dict(torch.load(f'{models_path}/critic/{opponent_path}'))

      # Run tournament between models
      wins = 0
      losses = 0
      print(f'playing as {model_path} vs {opponent_path}')
      for i in tqdm(range(round_robin_rematches)):
        reward, movelist, game = episode(
          model_actor, opponent_actor, model_critic, opponent_critic, hex_board_size, search_games_per_move, explore_coefficient, (i % 2) + 1, 'MCTS'
        )
        if reward == 1:
          wins += 1
        if reward == -1:
          losses += 1
      print(f'{wins} X {losses}')


if(__name__ == "__main__"):
  hex_board_size = config["hex_board_size"]
  search_games_per_move = config["search_games_per_move"]
  learning_rate = config["learning_rate"]
  weight_decay = config["weight_decay"]
  n_layers = config["n_layers"]
  kernel_size = config["kernel_size"]
  activation_function = config["activation_function"]
  skip_connection = config["skip_connection"]
  batch_size = config["batch_size"]
  name_optimizer = config["name_optimizer"]
  name_lossfunction = config["name_lossfunction"]
  n_models_to_save = config["n_models_to_save"]
  bootstrap_samples = config["bootstrap_samples"]
  episodes = config["episodes"]

  search_games_per_move = config["search_games_per_move"]
  round_robin_rematches = config["round_robin_rematches"]
  explore_coefficient = config["explore_coefficient"]

  model_save_path = train(
    hex_board_size,
    search_games_per_move,
    learning_rate,
    weight_decay,
    n_layers,
    kernel_size,
    activation_function,
    skip_connection,
    batch_size,
    name_optimizer,
    name_lossfunction,
    n_models_to_save,
    bootstrap_samples,
    episodes
  )
  topp(hex_board_size,
    search_games_per_move,
    round_robin_rematches,
    explore_coefficient,
    model_save_path
  )

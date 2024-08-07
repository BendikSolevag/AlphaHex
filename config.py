
config = {
  # • The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
  "hex_board_size": 4, # K, More than 2, less than 11.
  
  # • Standard MCTS parameters, such as the number of episodes, number of search games per actual move, etc.
  "episodes": 50,
  "search_games_per_move": 20,

  # • In the ANET, the learning rate, the number of hidden layers and neurons per layer, 
  # along with any of the following activation functions for hidden nodes: linear, sigmoid, tanh, RELU.
  "learning_rate": 0.01,
  "weight_decay": 0.000001,
  "n_layers": 2,
  "kernel_size": 3,
  "skip_connection": False,
  "activation_function": 'relu', # linear, sigmoid, tanh, relu
  "batch_size": 4,
  "explore_coefficient": 1,
  "bootstrap_samples": 20,

  # • The optimizer in the ANET, with (at least) the following options all available: 
  # Adagrad, Stochastic Gradient Descent (SGD), RMSProp, and Adam.
  "name_optimizer": "adam", # adagrad, sgd, rmsprop, adam
  "name_lossfunction": "mse",

  # • The number (M) of ANETs to be cached in preparation for a TOPP. These should be cached, starting with an 
  # untrained net prior to episode 1, at a fixed interval throughout the training episodes.
  "n_models_to_save": 3,

  # • The number of games, G, to be played between any two ANET-based agents that meet during the round-robin play of the TOPP.
  "round_robin_rematches": 10,
}


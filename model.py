import torch
import random
from torch import nn
from torch.optim import Adam, Adagrad, SGD, RMSprop
random.seed(42)

class ANET(nn.Module):
  def __init__(
      self, 
      model_depth: int, 
      input_channels: int,
      kernel_size: int,
      activation_function: str,
      skip_connection: bool
    ):
    super().__init__()
    self.skip_connection = skip_connection
    self.first_convolution = nn.Conv2d(input_channels, 3, kernel_size=kernel_size, padding=1)    
    self.convolutions = nn.ModuleList([nn.Conv2d(3, 3, kernel_size=kernel_size, padding=1) for _ in range(model_depth - 2)])
    self.last_convolution = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    self.flatten = nn.Flatten()
    self.softmax = nn.Softmax(dim=1)
    self.activation = nn.ReLU()
    if(activation_function == 'sigmoid'):
       self.activation = nn.Sigmoid()
    if(activation_function == 'tanh'):
       self.activation = nn.Tanh()
    if(activation_function == 'linear'):
       self.activation = lambda x: x

  def forward(self, x):
    # Scale input channels to convolution chain size
    x = self.activation(self.first_convolution(x))
    # Perform convolution chain
    for i in range(len(self.convolutions)):
       convolution = self.convolutions[i]
       residual = x
       x = convolution(x)
       x = self.activation(x)
       x = x + residual if self.skip_connection else x
    # Downscale to output size
    x = self.last_convolution(x)
    return self.softmax(self.flatten(x))
  

class CNET(nn.Module):
  def __init__(
      self, 
      model_depth: int, 
      input_channels: int,
      kernel_size: int,
      activation_function: str,
      skip_connection: bool
    ):
    super().__init__()
    self.skip_connection = skip_connection
    self.first_convolution = nn.Conv2d(input_channels, 3, kernel_size=kernel_size, padding=1)    
    self.last_convolution = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    self.linear = nn.Linear(16, 1)
    self.flatten = nn.Flatten()
    self.softmax = nn.Softmax(dim=1)
    self.activation = nn.ReLU()
    if(activation_function == 'sigmoid'):
       self.activation = nn.Sigmoid()
    if(activation_function == 'tanh'):
       self.activation = nn.Tanh()
    if(activation_function == 'linear'):
       self.activation = lambda x: x

  def forward(self, x):
    # Scale input channels to convolution chain size
    x = self.activation(self.first_convolution(x))
    # Perform convolution chain
    for i in range(len(self.convolutions)):
       convolution = self.convolutions[i]
       residual = x
       x = convolution(x)
       x = self.activation(x)
       x = x + residual if self.skip_connection else x
    # Downscale to output size
    x = self.activation(self.last_convolution(x))
    return self.linear(self.flatten(x))


def resolve_optimizer(params, optimizer, learning_rate, weight_decay) -> Adagrad | SGD | RMSprop | Adam:
  if(optimizer == 'adagrad'):
    return Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
  if(optimizer == 'sgd'):
    return SGD(params, lr=learning_rate, weight_decay=weight_decay)
  if(optimizer == 'rmsprop'):
    return RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
  if(optimizer == 'adam'):
    return Adam(params, lr=learning_rate, weight_decay=weight_decay)
  raise KeyError(f'Unable to resolve optimizer for input {optimizer}')


def resolve_lossfunction(lossfunction) -> nn.MSELoss:
  if lossfunction == 'mse':
    return nn.MSELoss()
  raise KeyError(f'Unable to resolve loss function for input {lossfunction}')


def backpropagate_anet(model: ANET, optimizer: Adagrad | SGD | RMSprop | Adam, lossfunc: nn.MSELoss, global_moves_list, batch_size: int, epochs: int):
  epochs = min(epochs, len(global_moves_list))
  for _ in range(epochs):
    optimizer.zero_grad()
    
    tmp = random.sample(global_moves_list, batch_size)
    modelinput = torch.cat([out[0] for out in tmp], 0)
    label = torch.cat([out[1] for out in tmp], 0)

    out = model.forward(modelinput)
    loss = lossfunc(out, label)
    loss.backward()
    optimizer.step()


def backpropagate_cnet(model: CNET, optimizer: Adagrad | SGD | RMSprop | Adam, lossfunc: nn.MSELoss, global_cnet_moves_list, batch_size: int, epochs: int):
  epochs = min(epochs, len(global_cnet_moves_list))
  for _ in range(epochs):
    optimizer.zero_grad()

    tmp = random.sample(global_cnet_moves_list, batch_size)
    modelinput = torch.cat([out[0] for out in tmp], 0)
    label = torch.tensor([[out[1]] for out in tmp])
    
    out = model.forward(modelinput)
    loss = lossfunc(out, label)
    loss.backward()
    optimizer.step()


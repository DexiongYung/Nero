import torch
import torch.nn as nn
import torch.nn.functional as F

from const import *


class Encoder(nn.Module):
    """
    Takes in an one-hot tensor of names and produces hidden state and cell state
    for decoder LSTM to use.

    input_size: N_LETTER
    hidden_size: Size of the hidden dimension
    """

    def __init__(self, input_size, hidden_size, embed_size: int = 8, num_layers=2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.to(DEVICE)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor = None):
        """
        Run LSTM through 1 time step.

        SHAPE REQUIREMENT
        - input: <1 x batch_size x EMBED_SZ>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        """
        if hidden is None:
            hidden = self.init_hidden(input.shape[1])

        input = self.embed(input)
        lstm_out, hidden = self.lstm(input, hidden)
        return lstm_out, hidden

    def init_hidden(self, batch_size: int):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))


class Decoder(nn.Module):
    """
    Accept hidden layers as an argument <num_layer x batch_size x hidden_size> for each hidden and cell state.
    At every forward call, output probability vector of <batch_size x output_size>.
    input_size: N_LETTER
    hidden_size: Size of the hidden dimension
    output_size: N_LETTER
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, embed_sz: int = 8, num_layers=2):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embed_sz = embed_sz

        # Initialize LSTM - Notice it does not accept output_size
        self.embed = nn.Embedding(input_size, embed_sz)
        self.lstm = nn.LSTM(embed_sz, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        self.to(DEVICE)

    def forward(self, input, hidden):
        """
        Run LSTM through 1 time step
        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        - lstm_out: <1 x batch_size x N_LETTER>
        """
        if hidden is None:
            hidden = self.init_hidden(input.shape[1])

        input = self.embed(input)
        lstm_out, hidden = self.lstm(input.unsqueeze(0), hidden)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.softmax(lstm_out)
        return lstm_out, hidden

    def init_hidden(self, batch_size: int):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))

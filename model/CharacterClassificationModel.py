import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from const import *
from utilities.infcomp_utilities import FORMAT_CLASS


class CharacterClassificationModel(nn.Module):
    def __init__(self, input: list, hidden_sz: int, output: list, num_layers: int = 4, embed_sz: int = 16,
                 drop_out: float = 0.1):
        """
        This module is made specifically for classifying characters in a name as first, middle, last, title 
        or suffix. Acts as a finite state automota by zeroing out moves that can't be taken
        """
        super(CharacterClassificationModel, self).__init__()
        self.input = input
        self.input_sz = len(input)
        self.hidden_sz = hidden_sz
        self.output = output
        self.output_sz = len(output)
        self.num_layers = num_layers
        self.embed_sz = embed_sz
        self.input_pad_idx = self.input.index(PAD)
        self.softmax = nn.Softmax(0)
        self.embed = nn.Embedding(self.input_sz, self.embed_sz, padding_idx=self.input_pad_idx)
        self.lstm = nn.LSTM(self.embed_sz, self.hidden_sz, num_layers=num_layers, bidirectional=True)
        self.fc1 = nn.Linear(self.hidden_sz * 2, self.hidden_sz)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(self.hidden_sz, self.output_sz)
        self.to(DEVICE)

    def forward(self, input: torch.LongTensor, address: str):
        embedded_input = self.embed(input).unsqueeze(1)
        length_tensor = torch.tensor([MAX_STRING_LEN]).to(DEVICE)
        pps_input = torch.nn.utils.rnn.pack_padded_sequence(embedded_input, length_tensor)

        X, hidden = self.lstm(pps_input, self.init_hidden())
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X)

        output = []
        sample = FORMAT_CLASS.index(SOS)
        for i in range(X.shape[0]):
            input = X[i].reshape(2 * self.hidden_sz)
            fc1_output = self.fc1.forward(input)
            activation_output = self.activation(fc1_output)
            fc2_output = self.fc2.forward(activation_output)
            probs = self.softmax(fc2_output)

            sample = int(pyro.sample(f"{address}_{i}", dist.Categorical(probs)).item())
            output.append(sample)

        return output

    def one_hot_encode(self, previous_sample: int):
        ret = torch.zeros(len(FORMAT_CLASS)).to(DEVICE)
        ret[previous_sample] = 1

        return ret

    def init_hidden(self, batch_sz: int = 1):
        return (torch.zeros(self.num_layers * 2, batch_sz, self.hidden_sz).to(DEVICE),
                torch.zeros(self.num_layers * 2, batch_sz, self.hidden_sz).to(DEVICE))

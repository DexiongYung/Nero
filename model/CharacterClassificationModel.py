import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from const import *


class CharacterClassificationModel(nn.Module):
    def __init__(self, input: list, hidden_sz: int, output: list, num_layers: int = 4, embed_sz: int = 16,
                 drop_out: float = 0.1):
        super(CharacterClassificationModel, self).__init__()
        self.input = input
        self.input_sz = len(input)
        self.hidden_sz = hidden_sz
        self.output = output
        self.output_sz = len(output)
        self.num_layers = num_layers
        self.embed_sz = embed_sz
        self.input_pad_idx = self.input.index(PAD)
        self.fc1_input_sz = self.num_layers * self.hidden_sz * 4

        self.softmax = nn.Softmax(0)
        self.fc1 = nn.Linear(self.fc1_input_sz, self.output_sz)
        self.embed = nn.Embedding(self.input_sz, self.embed_sz, padding_idx=self.input_pad_idx)
        self.forward_lstm = nn.LSTM(self.embed_sz, self.hidden_sz, num_layers=num_layers)
        self.backward_lstm = nn.LSTM(self.embed_sz, self.hidden_sz, num_layers=num_layers)
        self.to(DEVICE)

    def forward(self, input: torch.LongTensor, address: str):
        outputs = []
        input_len = len(input)
        hidden_states = []
        for i in range(input_len):
            hidden_states.append([])

        forward_hidden = self.init_hidden()
        backward_hidden = self.init_hidden()

        for i in range(input_len):
            forward_input = self.embed(input[i]).unsqueeze(0)
            _, forward_hidden = self.forward_lstm(forward_input.unsqueeze(0), forward_hidden)
            hidden_states[i].append(forward_hidden)

        for i in range(input_len):
            backward_idx = input_len - i - 1
            backward_input = self.embed(input[backward_idx]).unsqueeze(0)
            _, backward_hidden = self.backward_lstm(backward_input.unsqueeze(0), backward_hidden)
            hidden_states[backward_idx].append(backward_hidden)

        for i in range(input_len):
            hidden_1 = hidden_states[i][0]
            hidden_2 = hidden_states[i][1]
            hidden_1_cat = torch.cat((hidden_1[0], hidden_1[1]), 0)
            hidden_2_cat = torch.cat((hidden_2[0], hidden_2[1]), 0)
            hidden = torch.cat((hidden_1_cat, hidden_2_cat), 2)
            hidden = hidden.reshape(self.fc1_input_sz)
            scores = self.fc1.forward(hidden)
            probs = self.softmax(scores)
            outputs.append(int(pyro.sample(f"{address}_{i}", dist.Categorical(probs)).item()))

        return outputs

    def init_hidden(self, batch_sz: int = 1):
        return (torch.zeros(self.num_layers, batch_sz, self.hidden_sz).to(DEVICE),
                torch.zeros(self.num_layers, batch_sz, self.hidden_sz).to(DEVICE))

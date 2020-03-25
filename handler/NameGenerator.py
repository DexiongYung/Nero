import os
import torch

from const import *
from model.GenerativeModelLSTM import GenerativeModelLSTM
from utilities.config import *


class NameGenerator():
    def __init__(self, config_path: str, weights_path: str):
        super().__init__()
        config = load_json(config_path)
        self.hidden_sz = config['hidden_size']
        self.num_layers = config['num_layers']
        self.input = config['input']
        self.output = config['output']
        self.embed_sz = config['embed_dim']
        self.input_sz = len(self.input)
        self.output_sz = len(self.output)
        self.SOS = config['EOS']
        self.PAD = config['PAD']
        self.EOS = config['EOS']

        self.lstm = GenerativeModelLSTM(self.input_sz, self.hidden_sz, self.output_sz, self.embed_sz, self.num_layers)

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, path):
        if not os.path.exists(path):
            raise Exception(f"Path does not exist: {path}")
        self.lstm.load_state_dict(torch.load(path, map_location=DEVICE)['weights'])

    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor = None):
        with torch.no_grad():
            if hidden_state is None:
                hidden_state = self.lstm.initHidden(1)

            output, hidden = self.lstm.forward(input, hidden_state)
            return output, hidden

    def indexTensor(self, names: list, max_len: int):
        tensor = torch.zeros(max_len, len(names)).type(torch.LongTensor)
        for i, name in enumerate(names):
            for j, letter in enumerate(name):
                index = self.input.index(letter)

                if index < 0:
                    raise Exception(f'{names[j][i]} is not a char in {self.input}')

                tensor[j][i] = index
        return tensor.to(DEVICE)

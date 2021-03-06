import pyro
import pyro.distributions as dist
import string
import torch
import torch.nn as nn

from const import *
from model.seq2seq import Encoder, Decoder


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input: list, output: list, hidden_sz: int, num_layers: int, embed_dim: int = 8):
        super().__init__()
        self.input = input
        self.output = output
        self.input_sz = len(input)
        self.output_sz = len(output)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_sz = hidden_sz

        self.encoder = Encoder(self.input_sz, hidden_sz, num_layers=num_layers)
        self.decoder = Decoder(self.output_sz, hidden_sz,
                               self.output_sz, num_layers=num_layers)
        self.to(DEVICE)

    def forward(self, input: torch.Tensor, address: "str"):
        hidden = None

        for i in range(len(input)):
            printable_idx = input[i].item()
            printable_char = PRINTABLE[printable_idx]

            if printable_char not in string.ascii_letters + '\'-':
                continue

            _, hidden = self.encoder.forward(torch.LongTensor(
                [printable_idx]).unsqueeze(1).to(DEVICE), hidden)

        decoder_input = torch.LongTensor([self.output.index(PAD)]).to(DEVICE)
        outputs = []
        for i in range(MAX_OUTPUT_LEN):
            probs, hidden = self.decoder.forward(decoder_input, hidden)
            sample = int(pyro.sample(
                f"{address}_{i}", dist.Categorical(probs)).item())

            if sample == self.output.index(EOS):
                break

            outputs.append(sample)
            decoder_input = torch.LongTensor([sample]).to(DEVICE)

        return outputs

    def encode(self, input_str: str):
        input = torch.LongTensor([self.input.index(c)
                                  for c in input_str]).to(DEVICE)
        hidden = None
        for i in range(len(input)):
            printable_idx = input[i].item()
            printable_char = PRINTABLE[printable_idx]

            if printable_char not in string.ascii_letters + '\'-':
                continue

            _, hidden = self.encoder.forward(torch.LongTensor(
                [printable_idx]).unsqueeze(1).to(DEVICE), hidden)

        return hidden

    def decode(self, input_char: str, hidden: torch.Tensor):
        if input_char is None:
            input = torch.LongTensor([self.output.index(PAD)]).to(DEVICE)
        else:
            input = torch.LongTensor(
                [self.output.index(input_char)]).to(DEVICE)

        probs, hidden = self.decoder.forward(input, hidden)
        probs = torch.softmax(probs, dim=2)

        return probs, hidden

    def init_hidden(self, batch_sz: int):
        return (torch.zeros(self.num_layers, batch_sz, self.hidden_sz).to(DEVICE),
                torch.zeros(self.num_layers, batch_sz, self.hidden_sz).to(DEVICE))

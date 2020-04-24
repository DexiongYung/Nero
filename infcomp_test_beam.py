import argparse
import pyro
import torch
from const import *
from infcomp import NameParser, TITLE, SUFFIX
from utilities.config import load_json
from utilities.infcomp_utilities import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='filepath to config json',
                    type=str, default='config/UNNAMED_SESSION.json')
parser.add_argument('--name', help='name to parse',
                    nargs='?', default='Wood, Frunk', type=str)
parser.add_argument('--beam_width', help='Beam search width',
                    nargs='?', default=6, type=int)

args = parser.parse_args()
CONFIG = load_json(args.config)
NAME = args.name
BEAM_WIDTH = args.beam_width

rnn_hidden_size = CONFIG['rnn_hidden_size']
rnn_num_layer = CONFIG['rnn_num_layers']
format_hidden_size = CONFIG['format_hidden_size']

name_parser = NameParser(rnn_num_layer, rnn_hidden_size, format_hidden_size)
name_parser.load_checkpoint(filename=f"{CONFIG['session_name']}.pth.tar")

char_class_pad_idx = name_parser.guide_format.input_pad_idx

name_to_idx_lst = [PRINTABLE.index(
    c) for c in NAME] + [char_class_pad_idx] * (MAX_STRING_LEN - len(NAME))
input = torch.LongTensor(name_to_idx_lst).to(DEVICE)

char_classifications = name_parser.guide_format.forward(input, CHAR_FORMAT_ADD)
title, first, middle, last, suffix = parse_name(input, char_classifications)

fn_hidden = name_parser.guide_fn.encode(''.join(c for c in first[0]))
fn_topk = top_k_beam_search(name_parser.guide_fn, fn_hidden, BEAM_WIDTH)

print(fn_topk)

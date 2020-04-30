import argparse
import pyro
import torch
import pandas as pd
from const import *
from infcomp import NameParser, TITLE, SUFFIX
from utilities.config import load_json
from utilities.infcomp_utilities import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='filepath to config json',
                    type=str, default='config/UNNAMED_SESSION.json')
parser.add_argument('--name', help='name to parse',
                    nargs='?', default='Wood, Frank', type=str)
parser.add_argument('--beam_width', help='Beam search width',
                    nargs='?', default=6, type=int)

args = parser.parse_args()
CONFIG = load_json(args.config)
NAME = args.name
BEAM_WIDTH = args.beam_width

rnn_hidden_size = CONFIG['rnn_hidden_size']
rnn_num_layer = CONFIG['rnn_num_layers']
format_hidden_size = CONFIG['format_hidden_size']

name_parser = NameParser(
    rnn_num_layer, rnn_hidden_size, format_hidden_size)

name_parser.load_checkpoint(filename=f"{CONFIG['session_name']}.pth.tar")


def evaluate_name(name: str):
    char_class_pad_idx = name_parser.guide_format.input_pad_idx

    name_to_idx_lst = [PRINTABLE.index(
        c) for c in name] + [char_class_pad_idx] * (MAX_STRING_LEN - len(name))
    input = torch.LongTensor(name_to_idx_lst).to(DEVICE)

    char_classifications = name_parser.guide_format.forward(
        input, CHAR_FORMAT_ADD)
    title, first, middles, last, suffix = parse_name(
        input, char_classifications)

    fn_topk = get_topk_names(first, name_parser.guide_fn)
    mn_topk = get_topk_names(middles, name_parser.guide_fn)
    ln_topk = get_topk_names(last, name_parser.guide_ln)

    return fn_topk, mn_topk, ln_topk


def get_topk_names(name_list: list, autoencoder: DenoisingAutoEncoder):
    topk = []
    for name in name_list:
        hidden = autoencoder.encode(''.join(c for c in name))
        topk = top_k_beam_search(autoencoder, hidden, BEAM_WIDTH)

    return topk


def convert_to_str(list: list):
    name = ''

    for c in list:
        if c != EOS:
            name += c

    return name


def test_labelled_name_dataset(df: pd.DataFrame):
    """
    Takes a DataFrame and checks how many are correct when run through the beam search
    """
    f_correct, m_correct, l_correct, total = 0, 0, 0, 0

    for index, row in df.iterrows():
        total += 1
        full = row['name']
        first = row['first']
        middle = row['middle']
        last = row['last']

        fn_topk, mn_topk, ln_topk = evaluate_name(full)

        firsts = [convert_to_str(f) for f, _, _ in fn_topk]
        middles = [convert_to_str(m) for m, _, _ in mn_topk]
        lasts = [convert_to_str(l) for l, _, _ in ln_topk]

        if first in firsts:
            f_correct += 1

        if last in lasts:
            l_correct += 1

    return float(f_correct/total), float(l_correct/total)


df = pd.read_csv('data/british.csv')
print(test_labelled_name_dataset(df))

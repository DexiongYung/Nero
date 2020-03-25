import torch

from const import *


def letter_to_index(letter: str):
    if letter not in MODEL_CHARS: raise Exception(f"{letter} is not allowed!!!")
    return MODEL_CHARS.find(letter)


def printable_to_index(printable: str):
    if printable not in GUIDE_CHARS: raise Exception(f"{printable} is not allowed!!!")
    return GUIDE_CHARS.index(printable)


def pad_string(original: str, desired_len: int, pad_character: str = PAD, pre_pad: bool = False):
    """
    Returns the padded version of the original string to length: desired_len
    original: The string to be padded
    desired_len: The length of the new string
    pad_character: The character used to pad the original
    """
    if pre_pad:
        return (str(pad_character) * (desired_len - len(original))) + original
    else:
        return original + (str(pad_character) * (desired_len - len(original)))


def strings_to_tensor(strings: list, max_string_len: int, index_function):
    """
    Turn a list of name strings into a tensor of one-hot letter vectors
    of shape: <max_name_len x len(strings) x n_letters>

    All strings are padded with '<pad_character>' such that they have the length: desired_len
    strings: List of strings to converted to a one-hot-encded vector
    max_name_len: The max name length allowed
    """
    strings = list(map(lambda name: pad_string(name, max_string_len), strings))
    if index_function == letter_to_index:
        inner_dim = NUM_MODEL_CHARS
    else:
        inner_dim = NUM_GUIDE_CHARS
    tensor = torch.zeros(max_string_len, len(strings), inner_dim).to(DEVICE)
    for i_s, s in enumerate(strings):
        for i_char, letter in enumerate(s):
            tensor[i_char][i_s][index_function(letter)] = 1
    return tensor.to(DEVICE)


def strings_to_probs(strings: list, max_string_len: int, index_function, true_index_prob: float = 0.9) -> list:
    """
    Turn a list of strings into probabilities over rows where the element of the index
    of character has probability of 0.99 and others 0.01/(size(n_letters)-1)
    of shape: <max_string_len x batch_size (length of strings list) x n_letters>
    """
    strings = list(map(lambda name: pad_string(name, max_string_len), strings))
    if index_function == letter_to_index:
        inner_dim = NUM_MODEL_CHARS
    else:
        inner_dim = NUM_GUIDE_CHARS
    default_index_prob = (1. - true_index_prob) / inner_dim

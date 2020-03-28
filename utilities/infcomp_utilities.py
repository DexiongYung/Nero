import pyro
import pyro.distributions as dist
import re
import torch
from typing import Tuple

from const import *

FIRST_NAME_ADD = "first_name"
MIDDLE_NAME_ADD = "middle_name"
MIDDLE_FORMAT_ADD = "middle_name_format_id"
LAST_NAME_ADD = "last_name"
CHAR_NOISE_ADD = "char_noise"
CHAR_FORMAT_ADD = "char_format"
AUX_FORMAT_ADD = "aux_format_id"
MAIN_FORMAT_ADD = "main_format_id"
TITLE_ADD = "title"
SUFFIX_ADD = "suffix"

AUX_CLASS = ['{main}', '{title} {main}', '{main} {suffix}', '{title} {main} {suffix}']
MAIN_CLASS = ['{first} {last}', '{last}, {first}', '{first} {middle} {last}', '{last}, {first} {middle}']
MIDDLE_FORMAT_CLASS = ['{mn_init}.', '{mn_init}. {mn_init_1}.', '{mn_init}', '{mn_init} {mn_init_1}', '{mn}',
                       '{mn} {mn_1}']
TITLE = ['Mr', 'Mr.', 'Ms', 'Ms.', 'Mrs', 'Mrs.', 'Dr', 'Dr.', 'Sir', "Ma'am", 'Madam']
SUFFIX = ['Sr', 'Sr.', 'Snr', 'Jr', 'Jr.', 'Jnr', 'Phd', 'phd', 'md', 'MD', 'I', 'II', 'III', 'IV']
FORMAT_CLASS = ['t', 'f', 'm', 'l', 's', SPACE, DOT, COMMA, PAD, SOS]
# title, first, middle, last, suffix, 'separator', pad, SOS, EOS just for consistency. SOS is required for Transformer
NOISE_CLASS = ['n', 'a', 'r', 'd', PAD, SOS, EOS]
# none, add, replace, delete, PAD, SOS, EOS

AUX_FORMAT_DIM = len(AUX_CLASS)
MAIN_FORMAT_DIM = len(MAIN_CLASS)
MIDDLE_NAME_FORMAT_DIM = len(MIDDLE_FORMAT_CLASS)

MAIN_FORMAT_PROBS = torch.tensor([1 / MAIN_FORMAT_DIM] * MAIN_FORMAT_DIM).to(DEVICE)
AUX_FORMAT_PROBS = torch.tensor([1 / AUX_FORMAT_DIM] * AUX_FORMAT_DIM).to(DEVICE)
MIDDLE_NAME_FORMAT_PROBS = torch.tensor([1 / MIDDLE_NAME_FORMAT_DIM] * MIDDLE_NAME_FORMAT_DIM).to(DEVICE)
TITLE_PROBS = torch.tensor([1 / len(TITLE)] * len(TITLE)).to(DEVICE)
SUFFIX_PROBS = torch.tensor([1 / len(SUFFIX)] * len(SUFFIX)).to(DEVICE)


# No noise, add, replace, remove char


def generate_main_name(firstname: str, middlename: str, lastname: str, main_format_id: int,
                       middlename_char_format=None) -> Tuple[str, list]:
    has_middle = has_middle_name(main_format_id)
    full_name = MAIN_CLASS[main_format_id]
    char_format = full_name

    if has_middle and middlename_char_format is None:
        raise Exception(f"Is middle name format {main_format_id} but missing char format")

    if has_middle and middle_name_format is not None:
        full_name = full_name.format(first=firstname, middle=middlename, last=lastname)
        middle_name_format_str = "".join(c for c in middlename_char_format)
        char_format = char_format.format(first=len(firstname) * "f", middle=middle_name_format_str,
                                         last=len(lastname) * "l")
    else:
        full_name = full_name.format(first=firstname, last=lastname)
        char_format = char_format.format(first=len(firstname) * 'f', last=len(lastname) * 'l')

    char_format = [c for c in char_format]

    return full_name, char_format


def generate_aux_name(title: str, name: str, suffix: str, aux_format_id: int, main_char_format: list) -> Tuple[
    str, list]:
    is_title_format = has_title(aux_format_id)
    is_suffix_format = has_suffix(aux_format_id)

    if not is_suffix_format and not is_title_format:
        return name, main_char_format

    full_name = AUX_CLASS[aux_format_id]
    char_format = None

    if is_suffix_format and is_title_format:
        full_name = full_name.format(title=title, main=name, suffix=suffix)
        char_format = ['t'] * len(title) + [SPACE] + main_char_format + [SPACE] + ['s'] * len(suffix)
    elif is_suffix_format:
        full_name = full_name.format(main=name, suffix=suffix)
        char_format = main_char_format + [SPACE] + ['s'] * len(suffix)
    elif is_title_format:
        full_name = full_name.format(title=title, main=name)
        char_format = ['t'] * len(title) + [SPACE] + main_char_format

    return full_name, char_format


def sample_name(lstm, pyro_address_prefix) -> str:
    # Given a LSTM, generate a name
    name = ''
    input = lstm.indexTensor([[SOS]], 1)
    hidden = None
    for i in range(MAX_OUTPUT_LEN):
        output, hidden = lstm.forward(input[0], hidden)

        # This ensures no blank names are generated and first char is always capitalized
        if i == 0:
            output[0][0][lstm.output.index(EOS)] = -float('inf')

            for c in string.ascii_lowercase:
                output[0][0][lstm.output.index(c)] = -float('inf')

        output = torch.softmax(output, dim=2)[0][0]

        char_idx = int(pyro.sample(f"{pyro_address_prefix}_{i}", dist.Categorical(output)).item())
        character = lstm.output[char_idx]
        if char_idx is lstm.output.index(lstm.EOS):
            break
        input = lstm.indexTensor([[character]], 1)
        name += character
    return name


def parse_name(obs: torch.Tensor, classification: list):
    '''
    Parse name into components based on classification list, which classifies each obs
    index as first, middle, last, title, suffix, sep or pad
    '''
    class_str = ''.join(str(c) for c in classification)
    '''
    Separators can be noised to be non space characters so change them to
    space so can use .split in case there are multiple first, middle or last
    names
    '''
    for i in range(len(obs)):
        if classification[i] == 5:
            obs[i] = PRINTABLE.index(' ')

    titles, firsts, middles, lasts, suffixes = [], [], [], [], []

    for i in range(len(FORMAT_CLASS)):
        start = class_str.find(str(i))
        end = class_str.rfind(str(i))

        if start < 0 or end < 0:
            continue
        elif i == 0:
            titles = create_name_list(start, end, obs, classification, i)
        elif i == 1:
            firsts = create_name_list(start, end, obs, classification, i)
        elif i == 2:
            middles = create_name_list(start, end, obs, classification, i, True)
        elif i == 3:
            lasts = create_name_list(start, end, obs, classification, i)
        elif i == 4:
            suffixes = create_name_list(start, end, obs, classification, i)

    return titles, firsts, middles, lasts, suffixes


def create_name_list(start: int, end: int, obs: torch.Tensor, classification: list, class_idx: int,
                     multiple_allowed: bool = False):
    """
    Converts obs tensor to list of names(names are in list format so that PAD is considered 1 char in case of misclassification)
    Args:
        start: starting index of class
        end: ending index class
        obs: Observation tensor
        classification: list of each obs index classification
        class_idx: The classification id the index must have to append
        multiple_allowed: If multiple names should be allowed
    """
    ret = []
    name = []

    for n in range(start, end + 1):
        format_class = classification[n]

        if multiple_allowed and format_class == 5 and len(name) > 0:
            ret.append(name)
            name = []
        elif format_class == class_idx:
            index = obs[n].item()

            if index == PRINTABLE.index(PAD) or index == PRINTABLE.index(' '):
                continue

            name.append(PRINTABLE[index])

    if len(name) > 0:
        ret.append(name)

    return ret


def has_title(aux_format_id: int) -> bool:
    return '{title}' in AUX_CLASS[aux_format_id]


def has_suffix(aux_format_id: int) -> bool:
    return '{suffix}' in AUX_CLASS[aux_format_id]


def has_middle_name(main_format_id: int) -> bool:
    return '{middle}' in MAIN_CLASS[main_format_id]


def has_middle_initial(middle_name_format_id: int) -> bool:
    return '{mn_init}' in MIDDLE_FORMAT_CLASS[middle_name_format_id]


def num_middle_name(middle_name_format_id: int) -> int:
    return len(list(re.finditer(r'{.+?}', MIDDLE_FORMAT_CLASS[middle_name_format_id])))


def name_to_idx_tensor(name: list, allowed_chars: list, max_length: bool = False):
    '''
    Convert name in list where each index is a char to tensor form
    '''
    tensor_size = MAX_STRING_LEN if max_length else len(name)
    tensor = torch.zeros(tensor_size).type(torch.LongTensor)
    for i in range(tensor_size):
        if i < len(name):
            tensor[i] = allowed_chars.index(name[i])
        else:
            tensor[i] = allowed_chars.index(PAD)
    return tensor.to(DEVICE)


def classify_using_format_model(model, input: torch.tensor, unpadded_len: torch.IntTensor, address: str):
    '''
    Takes a list of lists classified as titles in input then classifies to 
    title or suffix in 'category' list
    '''
    probs = model(input, unpadded_len)
    return pyro.sample(address, dist.Categorical(probs.to(DEVICE))).item()


def generate_probabilities(string: str, categorical: list, peak_prob: float):
    string_length = len(string)
    categorical_length = len(categorical)
    probs = []

    for i in range(string_length):
        current_prob = [(1 - peak_prob) / (categorical_length - 1)] * categorical_length
        character_idx = categorical.index(string[i])
        current_prob[character_idx] = peak_prob
        probs.append(current_prob)

    return probs


def generate_main_name(main_name_format_id: int, first: str, last: str):
    main_name = MAIN_CLASS[main_name_format_id]
    has_middle = has_middle_name(main_name_format_id)

    if has_middle:
        main_name = main_name.format(first=first, last=last, middle='{middle}')
    else:
        main_name = main_name.format(first=first, last=last)

    return main_name


def generate_main_name_char_class(main_name_format_id: int, first: str, last: str):
    main_name = MAIN_CLASS[main_name_format_id]
    has_middle = has_middle_name(main_name_format_id)

    if has_middle:
        main_name = main_name.format(first=len(first) * 'f', last=len(last) * 'l', middle='{middle}')
    else:
        main_name = main_name.format(first=len(first) * 'f', last=len(last) * 'l')

    return main_name


def generate_aux_name(aux_format_id: int, title: str, suffix: str):
    is_title = has_title(aux_format_id)
    is_suffix = has_suffix(aux_format_id)
    aux_name = AUX_CLASS[aux_format_id]

    if is_title and is_suffix:
        aux_name = aux_name.format(title=title, suffix=suffix, main='{main}')
    elif is_title:
        aux_name = aux_name.format(title=title, main='{main}')
    elif is_suffix:
        aux_name = aux_name.format(suffix=suffix, main='{main}')

    return aux_name


def generate_aux_name_char_class(aux_format_id: int, title: str, suffix: str):
    is_title = has_title(aux_format_id)
    is_suffix = has_suffix(aux_format_id)
    aux_name = AUX_CLASS[aux_format_id]

    if is_title and is_suffix:
        aux_name = aux_name.format(title=len(title) * 't', suffix=len(suffix) * 's', main='{main}')
    elif is_title:
        aux_name = aux_name.format(title=len(title) * 't', main='{main}')
    elif is_suffix:
        aux_name = aux_name.format(suffix=len(suffix) * 's', main='{main}')

    return aux_name


def generate_middle_name(middle_name_format_id: int, middles: list):
    middle_name = MIDDLE_FORMAT_CLASS[middle_name_format_id]
    middle_count = num_middle_name(middle_name_format_id)
    is_initial_form = has_middle_initial(middle_name_format_id)

    if middle_count == 2:
        if is_initial_form:
            middle_name = middle_name.format(mn_init=middles[0], mn_init_1=middles[1])
        else:
            middle_name = middle_name.format(mn=middles[0], mn_1=middles[1])
    else:
        if is_initial_form:
            middle_name = middle_name.format(mn_init=middles[0])
        else:
            middle_name = middle_name.format(mn=middles[0])

    return middle_name


def generate_middle_name_char_class(middle_name_format_id: int, middles: list):
    middle_name = MIDDLE_FORMAT_CLASS[middle_name_format_id]
    middle_count = num_middle_name(middle_name_format_id)
    is_initial_form = has_middle_initial(middle_name_format_id)

    if middle_count == 2:
        if is_initial_form:
            middle_name = middle_name.format(mn_init=len(middles[0]) * 'm', mn_init_1=len(middles[1]) * 'm')
        else:
            middle_name = middle_name.format(mn=len(middles[0]) * 'm', mn_1=len(middles[1]) * 'm')
    else:
        if is_initial_form:
            middle_name = middle_name.format(mn_init=len(middles[0]) * 'm')
        else:
            middle_name = middle_name.format(mn=len(middles[0]) * 'm')

    return middle_name

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
TITLE_ADD = "title"
SUFFIX_ADD = "suffix"

AUX_CLASS = ['{main}', '{title} {main}', '{main} {suffix}', '{title} {main} {suffix}']
MAIN_CLASS = ['{first} {last}', '{last}, {first}', '{first} {middle} {last}', '{last}, {first} {middle}']
MIDDLE_FORMAT_CLASS = ['{mn_init}.', '{mn_init}. {mn_init1}.', '{mn_init}', '{mn_init} {mn_init1}', '{mn}',
                       '{mn} {mn1}']
TITLE = ['Mr', 'Mr.', 'Ms', 'Ms.', 'Mrs', 'Mrs.', 'Dr', 'Dr.', 'Sir', "Ma'am", 'Madam']
SUFFIX = ['Sr', 'Sr.', 'Snr', 'Jr', 'Jr.', 'Jnr', 'Phd', 'phd', 'md', 'MD', 'I', 'II', 'III', 'IV']
FORMAT_CLASS = ['t', 'f', 'm', 'l', 's', 'sep']
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
    char_format = []

    if has_middle:
        full_name = full_name.format(first=firstname, middle=middlename, last=lastname)
        char_format = ['sep'] * len(full_name)

        fn_start_idx = full_name.index(firstname)
        for i in range(len(firstname)):
            char_format[i + fn_start_idx] = FORMAT_CLASS[1]

        mn_start_idx = full_name.index(middlename)
        for i in range(len(middlename)):
            char_format[i + mn_start_idx] = middlename_char_format[i]

        ln_start_idx = full_name.index(lastname)
        for i in range(len(lastname)):
            char_format[i + ln_start_idx] = FORMAT_CLASS[3]

    else:
        full_name = full_name.format(first=firstname, last=lastname)
        char_format = ['sep'] * len(full_name)

        fn_start_idx = full_name.index(firstname)
        for i in range(len(firstname)):
            char_format[i + fn_start_idx] = FORMAT_CLASS[1]

        ln_start_idx = full_name.index(lastname)
        for i in range(len(lastname)):
            char_format[i + ln_start_idx] = FORMAT_CLASS[3]

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
        char_format = ['t'] * len(title) + ['sep'] + main_char_format + ['sep'] + ['s'] * len(suffix)
    elif is_suffix_format:
        full_name = full_name.format(main=name, suffix=suffix)
        char_format = main_char_format + ['sep'] + ['s'] * len(suffix)
    elif is_title_format:
        full_name = full_name.format(title=title, main=name)
        char_format = ['t'] * len(title) + ['sep'] + main_char_format

    return full_name, char_format


def middle_name_format(middlenames: list, middle_name_format_id: int) -> Tuple[str, list]:
    full_middle_name = ''
    character_classification = []
    is_middle_initial_format = has_middle_initial(middle_name_format_id)

    if num_middle_name(middle_name_format_id) == 2:
        middle_0 = middlenames[0]
        middle_1 = middlenames[1]
        if is_middle_initial_format:
            full_middle_name = MIDDLE_FORMAT_CLASS[middle_name_format_id].format(mn_init=middle_0, mn_init_1=middle_1)
        else:
            full_middle_name = MIDDLE_FORMAT_CLASS[middle_name_format_id].format(mn=middle_0, mn1=middle_1)
    else:
        middle = middlenames[0]
        if is_middle_initial_format:
            full_middle_name = MIDDLE_FORMAT_CLASS[middle_name_format_id].format(mn_init=middle)
        else:
            full_middle_name = MIDDLE_FORMAT_CLASS[middle_name_format_id].format(mn=middle)

    for c in full_middle_name:
        if c in string.ascii_letters + "\'-":
            character_classification.append('m')
        else:
            character_classification.append('sep')

    return full_middle_name, character_classification


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


def name_to_idx_tensor(name: list, allowed_chars: list):
    '''
    Convert name in list where each index is a char to tensor form
    '''
    tensor_size = len(name)
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


def generate_full_name_and_char_class(firstname: str, middlenames: list, lastname: str, main_format_id: int,
                                      middle_name_format_id: int) -> (torch.Tensor, torch.Tensor):
    main_name = MAIN_CLASS[main_format_id]
    main_name_char_class = main_name

    has_middle = has_middle_name(main_format_id)

    if has_middle:
        middle_name, middle_char_format = generate_middle_obs_and_char_probs(middlenames, middle_name_format_id)
        main_name = main_name.format(first=firstname, middle=middle_name, last=lastname)
        main_name_char_class = main_name_char_class.format(first="f" * len(firstname), middle=middle_char_format,
                                                           last="l" * len(lastname))
    else:
        main_name = main_name.format(first=firstname, last=lastname)
        main_name_char_class = main_name_char_class.format(first="f" * len(firstname), last="l" * len(lastname))
    
    if len(main_name) != len(main_name_char_class):
        raise Exception("Names are not the same length")

    return main_name, main_name_char_class


def generate_obs_and_char_probs(main_name: str, char_classes: str, peak_prob: float):
    character_format_class_probs = []
    observation_probs = []

    for i in range(len(main_name)):
        char_prob = [(1 - peak_prob) / len(FORMAT_CLASS)] * len(FORMAT_CLASS)
        observation_prob = [(1 - peak_prob) / len(PRINTABLE)] * len(PRINTABLE)

        if char_classes[i] in FORMAT_CLASS:
            index = FORMAT_CLASS.index(char_classes[i])
            char_prob[index] = peak_prob
        else:
            char_prob[5] = peak_prob

        curr_letter_idx = PRINTABLE.index(main_name[i])
        observation_prob[curr_letter_idx] = peak_prob

        character_format_class_probs.append(char_prob)
        observation_probs.append(observation_prob)

    return observation_probs, character_format_class_probs


def generate_middle_obs_and_char_probs(middlenames: list, middle_name_format_id: int):
    full_middle = MIDDLE_FORMAT_CLASS[middle_name_format_id]
    middle_char_class = full_middle

    if num_middle_name(middle_name_format_id) == 2:
        middle_0 = middlenames[0]
        middle_1 = middlenames[1]

        if has_middle_initial(middle_name_format_id):
            middle_char_class = middle_char_class.format(mn_init="m" * len(middle_0), mn_init1="m" * len(middle_1))
            full_middle = full_middle.format(mn_init=middle_0, mn_init1=middle_1)
        else:
            middle_char_class = middle_char_class.format(mn="m" * len(middle_0), mn1="m" * len(middle_1))
            full_middle = full_middle.format(mn=middle_0, mn1=middle_1)
    else:
        middle = middlenames[0]

        if has_middle_initial(middle_name_format_id):
            middle_char_class = middle_char_class.format(mn_init="m" * len(middle))
            full_middle = full_middle.format(mn_init=middle)
        else:
            middle_char_class = middle_char_class.format(mn="m" * len(middle))
            full_middle = full_middle.format(mn=middle)

    return full_middle, middle_char_class

import math
from random import randint

CHARACTER_REPLACEMENT = dict()
CHARACTER_REPLACEMENT['A'] = 'QSZWXa'
CHARACTER_REPLACEMENT['B'] = 'NHGVb '
CHARACTER_REPLACEMENT['C'] = 'VFDXc '
CHARACTER_REPLACEMENT['D'] = 'FRESXCd'
CHARACTER_REPLACEMENT['E'] = 'SDFR$#WSe'
CHARACTER_REPLACEMENT['F'] = 'GTRDCVf'
CHARACTER_REPLACEMENT['G'] = 'HYTFVBg'
CHARACTER_REPLACEMENT['H'] = 'JUYTGBNh'
CHARACTER_REPLACEMENT['I'] = 'UJKLO(*i'
CHARACTER_REPLACEMENT['J'] = 'MKIUYHNj'
CHARACTER_REPLACEMENT['K'] = 'JM<LOIk'
CHARACTER_REPLACEMENT['L'] = 'K<>:POl'
CHARACTER_REPLACEMENT['M'] = 'NJK< m'
CHARACTER_REPLACEMENT['N'] = 'BHJM n'
CHARACTER_REPLACEMENT['O'] = 'PLKI()Po'
CHARACTER_REPLACEMENT['P'] = 'OL:{_)O"p'
CHARACTER_REPLACEMENT['Q'] = 'ASW@!q'
CHARACTER_REPLACEMENT['R'] = 'TFDE$r%'
CHARACTER_REPLACEMENT['S'] = 'DXZAWEs'
CHARACTER_REPLACEMENT['T'] = 'YGFR%^t'
CHARACTER_REPLACEMENT['U'] = 'IJHY&*u'
CHARACTER_REPLACEMENT['V'] = 'CFGB v'
CHARACTER_REPLACEMENT['W'] = 'SAQ@#Ew'
CHARACTER_REPLACEMENT['X'] = 'ZASDCx'
CHARACTER_REPLACEMENT['Y'] = 'UGHT^&y'
CHARACTER_REPLACEMENT['Z'] = 'XSAz'
CHARACTER_REPLACEMENT['a'] = 'qwszA'
CHARACTER_REPLACEMENT['b'] = 'nhgv B'
CHARACTER_REPLACEMENT['c'] = 'vfdx C'
CHARACTER_REPLACEMENT['d'] = 'fresxcD'
CHARACTER_REPLACEMENT['e'] = 'sdfr43wsE'
CHARACTER_REPLACEMENT['f'] = 'gtrdcvF'
CHARACTER_REPLACEMENT['g'] = 'hytfvbG'
CHARACTER_REPLACEMENT['h'] = 'juytgbnH'
CHARACTER_REPLACEMENT['i'] = 'ujklo98I'
CHARACTER_REPLACEMENT['j'] = 'mkiuyhnJ'
CHARACTER_REPLACEMENT['k'] = 'jm,loijK'
CHARACTER_REPLACEMENT['l'] = 'k,.;pokL'
CHARACTER_REPLACEMENT['m'] = 'njk, M'
CHARACTER_REPLACEMENT['n'] = 'bhjm N'
CHARACTER_REPLACEMENT['o'] = 'plki90pO'
CHARACTER_REPLACEMENT['p'] = 'ol;[-0oP'
CHARACTER_REPLACEMENT['q'] = 'asw21Q'
CHARACTER_REPLACEMENT['r'] = 'tfde45R'
CHARACTER_REPLACEMENT['s'] = 'dxzaweS'
CHARACTER_REPLACEMENT['t'] = 'ygfr56T'
CHARACTER_REPLACEMENT['u'] = 'ijhy78U'
CHARACTER_REPLACEMENT['v'] = 'cfgb V'
CHARACTER_REPLACEMENT['w'] = 'saq23eW'
CHARACTER_REPLACEMENT['x'] = 'zsdcX'
CHARACTER_REPLACEMENT['y'] = 'uhgt67Y'
CHARACTER_REPLACEMENT['z'] = 'xsaZ'
CHARACTER_REPLACEMENT['1'] = '2q'
CHARACTER_REPLACEMENT['2'] = '3wq1'
CHARACTER_REPLACEMENT['3'] = '4ew2'
CHARACTER_REPLACEMENT['4'] = '5re3'
CHARACTER_REPLACEMENT['5'] = '6tr4'
CHARACTER_REPLACEMENT['6'] = '7yt5'
CHARACTER_REPLACEMENT['7'] = '8uy6'
CHARACTER_REPLACEMENT['8'] = '9iu7'
CHARACTER_REPLACEMENT['9'] = '0oi8'
CHARACTER_REPLACEMENT['0'] = '-po9'


def noise_list(input: list, allowed_chars: str, max_len: int, max_noise: int = 2):
    ret = []
    for i in range(len(input)):
        ret.append(noise_name(input[i], allowed_chars, max_len, max_noise))
    return ret


def noise_name(x: str, allowed_chars: str, max_length: int, max_noise: int = 2):
    noise_type = randint(0, 3)

    if noise_type == 0:
        return add_chars(x, allowed_chars, max_length, max_add=max_noise)
    elif noise_type == 1:
        return switch_chars(x, allowed_chars, max_switch=max_noise)
    elif 2:
        return remove_chars(x, max_remove=max_noise)
    else:
        x = remove_chars(x, max_remove=max_noise)
        return add_chars(x, allowed_chars, max_add=max_noise)


def add_chars(x: str, allowed_chars: str, max_length: int, max_add: int):
    if max_add + len(x) > max_length:
        raise Exception(f"{max_add + len(x)} is greater than max length:{max_length}")

    ret = x
    num_to_add = randint(0, max_add)

    for i in range(num_to_add):
        random_char = allowed_chars[randint(0, len(allowed_chars) - 1)]
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], random_char, ret[pos:]))

    return ret


def switch_chars(x: str, allowed_chars: str, max_switch: int):
    ret = x
    num_to_switch = randint(0, min(math.floor(len(x) / 2), max_switch))

    for i in range(num_to_switch):
        random_char = allowed_chars[randint(0, len(allowed_chars) - 1)]
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], random_char, ret[pos + 1:]))

    return ret


def remove_chars(x: str, max_remove: int):
    ret = x
    num_to_remove = randint(0, min(math.floor(len(x) / 2), max_remove))

    for i in range(num_to_remove):
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], ret[pos + 1:]))

    return ret

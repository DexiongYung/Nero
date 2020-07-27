import torch

# Levenshtein distance between clean and noisy word given the length (length is at the end of name
distance = {
    'lev_dist_1': [0.4533, 0.5422, 0.0042, 0.0, 0.0, 0.0, 0.0],
    'lev_dist_2': [0.8052, 0.1844, 0.0081, 0.0016, 0.0004, 0.0003, 0.0],
    'lev_dist_3': [0.7444, 0.2145, 0.0339, 0.0046, 0.0013, 0.0006, 0.0006],
    'lev_dist_4': [0.6628, 0.2792, 0.0421, 0.0129, 0.002, 0.0007, 0.0002],
    'lev_dist_5': [0.6928, 0.2378, 0.0473, 0.0161, 0.0047, 0.0009, 0.0003],
    'lev_dist_6': [0.6954, 0.2129, 0.058, 0.0228, 0.0081, 0.002, 0.0007],
    'lev_dist_7': [0.6867, 0.2057, 0.0585, 0.0291, 0.0137, 0.0048, 0.0012],
    'lev_dist_8': [0.6087, 0.2324, 0.0784, 0.0451, 0.022, 0.0094, 0.0034],
    'lev_dist_9': [0.6025, 0.2053, 0.0857, 0.0537, 0.0287, 0.0155, 0.0082],
    'lev_dist_10': [0.5337, 0.2179, 0.109, 0.0655, 0.0392, 0.0218, 0.0121],
    'lev_dist_11': [0.5588, 0.1958, 0.1035, 0.0693, 0.0372, 0.0237, 0.0111],
    'lev_dist_12': [0.4281, 0.2432, 0.1418, 0.0893, 0.0509, 0.0296, 0.0168],
    'lev_dist_13': [0.4592, 0.1926, 0.1171, 0.0926, 0.0705, 0.0401, 0.0277],
    'lev_dist_14': [0.5301, 0.208, 0.1142, 0.0651, 0.0394, 0.0292, 0.0141],
    'lev_dist_15': [0.3894, 0.2864, 0.1636, 0.0864, 0.0394, 0.0152, 0.0167],
    'lev_dist_16': [0.8829, 0.0878, 0.0049, 0.0146, 0.0098, 0.0, 0.0],
    'lev_dist_17': [0.9024, 0.0732, 0.0244, 0.0, 0.0, 0.0, 0.0],
    'lev_dist_18': [0.7609, 0.1739, 0.0, 0.0435, 0.0, 0.0, 0.0217],
    'lev_dist_19': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# Edit type (insertion, deletion, substitutions distribution given length
edit = {
    "edit_cate_1":
    [0.6440632538193514, 0.0, 0.35593674618064863],
    "edit_cate_2":
    [0.26924446343373926, 0.02271594388663391, 0.7080395926796269],
    "edit_cate_3":
    [0.21225766835287582, 0.24605982631071083, 0.5416825053364134],
    "edit_cate_4":
    [0.2570691747572815, 0.20766080097087378, 0.5352700242718447],
    "edit_cate_5":
    [0.2555279881605293, 0.2404130756507356, 0.5040589361887351],
    "edit_cate_6":
    [0.21157256480232767, 0.2698920049696828, 0.5185354302279895],
    "edit_cate_7":
    [0.1800266289560953, 0.2690847743568115, 0.5508885966870932],
    "edit_cate_8":
    [0.16017893007120687, 0.2785466496257075, 0.5612744203030856],
    "edit_cate_9":
    [0.15604391123976882, 0.2825384529136555, 0.5614176358465757],
    "edit_cate_10":
    [0.13757361891529676, 0.28075395605611636, 0.5816724250285868],
    "edit_cate_11":
    [0.12153567615245346, 0.32793277935100507, 0.5505315444965415],
    "edit_cate_12":
    [0.09600997506234414, 0.3031131847799855, 0.6008768401576703],
    "edit_cate_13":
    [0.08923916338284328, 0.3617459836314035, 0.5490148529857533],
    "edit_cate_14":
    [0.11187520966118752, 0.35575310298557533, 0.5323716873532371],
    "edit_cate_15":
    [0.06349206349206349, 0.35361552028218696, 0.5828924162257496],
    "edit_cate_16":
    [0.05090311986863711, 0.2348111658456486, 0.7142857142857143],
    "edit_cate_17":
    [0.0, 0.8478260869565217, 0.15217391304347827],
    "edit_cate_18":
    [0.09090909090909091, 0.8939393939393939, 0.015151515151515152]
}

alpha_os_word_perc = 0.3262333403736625
lower_vowel_os_word_perc = 0.1371845033978279
upper_vowel_os_word_perc = 0.005717662437661721
lower_consonants_os_word_perc = 0.1708350867140005
upper_consonants_os_word_perc = 0.012636875609952636
punc_os_word_perc = 0.009465066244724228
white_sp_os_word_perc = 0.0042796848319091585
digit_os_word_perc = 0.0001396133069380654
char_within_word_perc = 1 - alpha_os_word_perc - \
    punc_os_word_perc - white_sp_os_word_perc - digit_os_word_perc
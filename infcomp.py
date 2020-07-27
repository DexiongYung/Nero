import os
import pyro
import pyro.distributions as dist
import string
import torch

from const import *
from handler.NameGenerator import NameGenerator
from model.CharacterClassificationModel import CharacterClassificationModel
from model.DenoisingAutoEncoder import DenoisingAutoEncoder
from model.FormatModel import NameFormatModel
from model.NameCharacterClassifierModel import NameCharacterClassifierModel
from utilities.config import *
from utilities.infcomp_utilities import *
from utilities.Prob_Prog_Model import noise_name


class NameParser():
    """
    Generates names using a separate LSTM for first, middle, last name and a neural net
    using ELBO to parameterize NN for format classification.

    input_size: Should be the number of letters to allow
    hidden_size: Size of the hidden dimension in LSTM
    num_layers: Number of hidden layers in LSTM
    hidden_sz: Hidden layer size for LSTM RNN
    peak_prob: The max expected probability
    """

    def __init__(self, num_layers: int, hidden_sz: int, format_hidden_sz: int, peak_prob: float = 0.99):
        super().__init__()
        # Load up BART output vocab to correlate with name generative models.
        config = load_json('config/first.json')
        self.output_chars = config['output']
        self.num_output_chars = len(self.output_chars)
        # Model neural nets instantiation
        self.model_fn = NameGenerator(
            'config/first.json', 'nn_model/first.path.tar')
        self.model_ln = NameGenerator(
            'config/last.json', 'nn_model/last.path.tar')
        # Guide neural nets instantiation
        """
        Output for pretrained LSTMS doesn't have SOS so just use PAD for SOS
        """
        self.guide_fn = DenoisingAutoEncoder(
            PRINTABLE, self.output_chars, hidden_sz, num_layers)
        self.guide_ln = DenoisingAutoEncoder(
            PRINTABLE, self.output_chars, hidden_sz, num_layers)
        # Format classifier neural networks
        self.guide_format = NameCharacterClassifierModel(
            PRINTABLE, hidden_sz, FORMAT_CLASS)
        # Title / Suffix classifier neural networks
        self.guide_title = NameFormatModel(
            PRINTABLE, hidden_sz=hidden_sz, output_sz=len(TITLE))
        self.guide_suffix = NameFormatModel(
            PRINTABLE, hidden_sz=hidden_sz, output_sz=len(SUFFIX))
        self.guide_aux_format = NameFormatModel(PRINTABLE, hidden_sz=format_hidden_sz,
                                                output_sz=AUX_FORMAT_DIM)
        self.guide_main_format = NameFormatModel(PRINTABLE, hidden_sz=format_hidden_sz,
                                                 output_sz=MAIN_FORMAT_DIM)
        self.guide_mn_format = NameFormatModel(PRINTABLE, hidden_sz=format_hidden_sz,
                                               output_sz=MIDDLE_NAME_FORMAT_DIM)
        # Hyperparameters
        self.peak_prob = peak_prob
        self.num_layers = num_layers
        self.hidden_sz = hidden_sz

    def model(self, observations={"output": 0}):
        with torch.no_grad():
            # Sample format
            aux_format_id = int(pyro.sample(
                AUX_FORMAT_ADD, dist.Categorical(AUX_FORMAT_PROBS)).item())
            main_format_id = int(pyro.sample(
                MAIN_FORMAT_ADD, dist.Categorical(MAIN_FORMAT_PROBS)).item())

            # Sample title, first name, middle name, last name, and/or suffix
            firstname, middlename, lastname, title, suffix = None, None, None, None, None
            has_middle = has_middle_name(main_format_id)
            is_title = has_title(aux_format_id)
            is_suffix = has_suffix(aux_format_id)

            if is_title:
                title = TITLE[int(pyro.sample(
                    TITLE_ADD, dist.Categorical(TITLE_PROBS)).item())]

            if is_suffix:
                suffix = SUFFIX[int(pyro.sample(
                    SUFFIX_ADD, dist.Categorical(SUFFIX_PROBS)).item())]

            # first & last name generation
            firstname = sample_name(self.model_fn, FIRST_NAME_ADD)
            lastname = sample_name(self.model_ln, LAST_NAME_ADD)

            # Middle Name generation
            middle_name_format_id = None

            if has_middle:
                middle_name_format_id = int(
                    pyro.sample(MIDDLE_FORMAT_ADD, dist.Categorical(MIDDLE_NAME_FORMAT_PROBS)).item())
                middlenames = []

                for i in range(num_middle_name(middle_name_format_id)):
                    if has_middle_initial(middle_name_format_id):
                        initial_probs = torch.zeros(
                            self.num_output_chars).to(DEVICE)
                        initial_probs[26:52] = 1 / 26
                        letter_idx = int(
                            pyro.sample(f"{MIDDLE_NAME_ADD}_{i}_0", dist.Categorical(initial_probs)).item())
                        initial_format_probs = torch.zeros(
                            self.num_output_chars).to(DEVICE)
                        initial_format_probs[-2] = 1.
                        pyro.sample(f"{MIDDLE_NAME_ADD}_{i}_1",
                                    dist.Categorical(initial_format_probs))
                        # For capital names
                        middlename = self.output_chars[letter_idx]
                    else:
                        middlename = sample_name(
                            self.model_fn, f"{MIDDLE_NAME_ADD}_{i}")
                    middlenames.append(middlename)

            allowed_noise = [c for c in string.ascii_letters + string.digits]
            noised_first = noise_name(firstname)
            noised_last = noise_name(lastname)
            noised_title, noised_suffix = None, None

            if is_suffix:
                noised_suffix = noise_name(suffix)

            if is_title:
                noised_title = noise_name(title)

            main_section = generate_main_name(
                main_format_id, noised_first, noised_last)
            main_section_char_class = generate_main_name_char_class(
                main_format_id, noised_first, noised_last)

            aux_section = generate_aux_name(
                aux_format_id, noised_title, noised_suffix)
            aux_section_char_class = generate_aux_name_char_class(
                aux_format_id, noised_title, noised_suffix)

            full_name = aux_section.replace('{main}', main_section)
            character_classes = aux_section_char_class.replace(
                '{main}', main_section_char_class)

            noised_middles = []
            if has_middle:
                for i in range(len(middlenames)):
                    noised_middles.append(noise_name(
                        middlenames[i]))
                middle_section = generate_middle_name(
                    middle_name_format_id, noised_middles)
                middle_char_class = generate_middle_name_char_class(
                    middle_name_format_id, noised_middles)

                full_name = full_name.replace('{middle}', middle_section)
                character_classes = character_classes.replace(
                    '{middle}', middle_char_class)

            if len(full_name) != len(character_classes):
                raise Exception(
                    "Full name and character classes aren't the same length")
            elif len(full_name) < MAX_STRING_LEN:
                full_name = [c for c in full_name] + \
                    [PAD] * (MAX_STRING_LEN - len(full_name))
                character_classes = [c for c in character_classes] + \
                    [PAD] * (MAX_STRING_LEN - len(character_classes))

            character_probs = generate_probabilities(
                character_classes, FORMAT_CLASS, self.peak_prob)
            observation_probs = generate_probabilities(
                full_name, PRINTABLE, self.peak_prob)

            format_samples = []
            for i in range(MAX_STRING_LEN):
                format_samples.append(pyro.sample(f"{CHAR_FORMAT_ADD}_{i}",
                                                  dist.Categorical(torch.tensor(character_probs[i]).to(DEVICE))).item())

            observation_samples = pyro.sample("output",
                                              dist.Categorical(
                                                  torch.tensor(observation_probs[:MAX_STRING_LEN]).to(DEVICE)),
                                              obs=observations["output"])

        parse = {'firstname': firstname, 'middlename': middlename, 'lastname': lastname, 'title': title,
                 'suffix': suffix}

        print("full name: {}, format probs: {}".format(
            "".join(PRINTABLE[observation_samples[i].item()]
                    for i in range(len(observation_samples))).replace(PAD, ''),
            "".join(FORMAT_CLASS[format_samples[i]] for i in range(len(format_samples))).replace(PAD, '')))

        print(
            "first name: {}, middle name: {}, last name: {}, title: {}, suffix: {}, middle name format: {}, main format: {}, aux format: {} \n".format(
                firstname, middlename, lastname, title, suffix, middle_name_format_id, main_format_id, aux_format_id))

        return full_name, parse

    def guide(self, observations=None):
        X = observations['output']
        X_len = len(X)
        X_unpadded_len = X_len - (X == PRINTABLE.index(PAD)).sum(dim=0).item()
        X_unpadded_tensor = torch.IntTensor([X_unpadded_len]).to(DEVICE)

        # Infer formats and parse
        pyro.module("format_lstm", self.guide_format.lstm)
        pyro.module("format_fc1", self.guide_format.fc1)
        pyro.module("format_fc2", self.guide_format.fc2)
        char_class_samples = self.guide_format.forward(X, CHAR_FORMAT_ADD)

        title, firsts, middles, lasts, suffix = parse_name(
            X, char_class_samples)
        cleaned_firsts, cleaned_middles, cleaned_lasts = [], [], []

        pyro.module("aux_format", self.guide_aux_format)
        classify_using_format_model(
            self.guide_aux_format, X, X_unpadded_tensor, AUX_FORMAT_ADD)

        pyro.module("main_format", self.guide_main_format)
        classify_using_format_model(
            self.guide_main_format, X, X_unpadded_tensor, MAIN_FORMAT_ADD)

        if len(title) > 0:
            pyro.module("title", self.guide_title)
            title_tensor = name_to_idx_tensor(title[0], PRINTABLE)
            sample = classify_using_format_model(self.guide_title, title_tensor,
                                                 torch.IntTensor([len(title_tensor)]).to(DEVICE), TITLE_ADD)
            title = TITLE[sample]

        if len(suffix) > 0:
            pyro.module("suffix", self.guide_suffix)
            suffix_tensor = name_to_idx_tensor(suffix[0], PRINTABLE)
            sample = classify_using_format_model(self.guide_suffix, suffix_tensor,
                                                 torch.IntTensor([len(suffix_tensor)]).to(DEVICE), SUFFIX_ADD)
            suffix = SUFFIX[sample]

        for first in firsts:
            pyro.module("first_name", self.guide_fn)
            input = name_to_idx_tensor(first, self.guide_fn.input)
            samples = self.guide_fn.forward(input, FIRST_NAME_ADD)
            cleaned_firsts.append(
                ''.join(self.output_chars[s] for s in samples))

        if len(middles) > 0:
            pyro.module("first_name", self.guide_fn)
            pyro.module("middle_name_format", self.guide_mn_format)
            classify_using_format_model(self.guide_mn_format, X, torch.IntTensor([len(X)]).to(DEVICE),
                                        MIDDLE_FORMAT_ADD)

            for i in range(len(middles)):
                middle = middles[i]
                input = name_to_idx_tensor(middle, self.guide_fn.input)
                samples = self.guide_fn.forward(
                    input, f"{MIDDLE_NAME_ADD}_{i}")
                cleaned_middles.append(
                    ''.join(self.output_chars[s] for s in samples))

        for last in lasts:
            pyro.module("last_name", self.guide_ln)
            input = name_to_idx_tensor(last, self.guide_ln.input)
            samples = self.guide_ln.forward(input, LAST_NAME_ADD)
            cleaned_lasts.append(
                ''.join(self.output_chars[s] for s in samples))

        # TODO!!! Have to add full name reconstruction

        return {'firstname': cleaned_firsts, 'middlename': cleaned_middles, 'lastname': cleaned_lasts, 'title': title, 'suffix': suffix}

    def infer(self, names: list):
        # Infer using q(z|x)
        results = []
        for name in names:
            encoded_name = self.get_observes(name)
            result = self.guide(observations={'output': encoded_name})
            results.append(result)
        return results

    def generate(self, num_samples: int = 1):
        # Generate samples from p(x,z)
        results = []
        for _ in range(num_samples):
            results.append(self.model()[0])
        return results

    def test_mode(self):
        self.guide_format.eval()
        self.guide_aux_format.eval()
        self.guide_main_format.eval()
        self.guide_mn_format.eval()
        self.guide_title.eval()
        self.guide_suffix.eval()

    def get_observes(self, name_string: str):
        if len(name_string) > MAX_STRING_LEN:
            raise Exception(
                f"Name string length cannot exceed {MAX_STRING_LEN}.")
        name_as_list = [c for c in name_string]
        return name_to_idx_tensor(name_as_list, PRINTABLE, max_length=True)

    def load_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        name_fp = os.path.join(folder, "name_" + filename)
        format_fp = os.path.join(folder, "format_" + filename)
        title_suffix_fp = os.path.join(folder, "title_suffix_" + filename)

        if not os.path.exists(name_fp) or not os.path.exists(format_fp) or not os.path.exists(title_suffix_fp):
            raise Exception(f"No model in path {folder}")

        name_content = torch.load(name_fp, map_location=DEVICE)
        format_content = torch.load(format_fp, map_location=DEVICE)
        title_suffix_content = torch.load(title_suffix_fp, map_location=DEVICE)

        # name content
        self.guide_fn.load_state_dict(name_content['guide_fn'])
        self.guide_ln.load_state_dict(name_content['guide_ln'])
        # title and suffix
        self.guide_title.load_state_dict(title_suffix_content['title'])
        self.guide_suffix.load_state_dict(title_suffix_content['suffix'])
        # format content
        self.guide_format.load_state_dict(format_content['guide_format'])
        self.guide_aux_format.load_state_dict(format_content['aux_format'])
        self.guide_main_format.load_state_dict(format_content['main_format'])
        self.guide_mn_format.load_state_dict(
            format_content['middle_name_format'])

    def save_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        name_fp = os.path.join(folder, "name_" + filename)
        title_suffix_fp = os.path.join(folder, "title_suffix_" + filename)
        format_fp = os.path.join(folder, "format_" + filename)

        if not os.path.exists(folder):
            os.mkdir(folder)

        name_content = {
            'guide_fn': self.guide_fn.state_dict(),
            'guide_ln': self.guide_ln.state_dict(),
        }
        title_suffix_content = {
            'title': self.guide_title.state_dict(),
            'suffix': self.guide_suffix.state_dict(),
        }
        format_content = {
            'guide_format': self.guide_format.state_dict(),
            'aux_format': self.guide_aux_format.state_dict(),
            'main_format': self.guide_main_format.state_dict(),
            'middle_name_format': self.guide_mn_format.state_dict(),
        }
        torch.save(name_content, name_fp)
        torch.save(format_content, format_fp)
        torch.save(title_suffix_content, title_suffix_fp)

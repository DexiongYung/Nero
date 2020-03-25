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
from utilities.noiser import *


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

    def __init__(self, num_layers: int = 2, hidden_sz: int = 64, peak_prob: float = 0.999, format_hidd_sz: int = 64,
                 noise_prob: int = 0.10):
        super().__init__()
        # Load up BART output vocab to correlate with name generative models.
        config = load_json('config/first.json')
        self.output_chars = config['output']
        self.num_output_chars = len(self.output_chars)
        # Model neural nets instantiation
        self.model_fn = NameGenerator('config/first.json', 'nn_model/first.path.tar')
        self.model_ln = NameGenerator('config/last.json', 'nn_model/last.path.tar')
        # Guide neural nets instantiation
        """
        Output for pretrained LSTMS doesn't have SOS so just use PAD for SOS
        """
        self.guide_fn = DenoisingAutoEncoder(PRINTABLE, self.output_chars, hidden_sz, num_layers)
        self.guide_ln = DenoisingAutoEncoder(PRINTABLE, self.output_chars, hidden_sz, num_layers)
        # Format classifier neural networks
        self.guide_format = NameCharacterClassifierModel(PRINTABLE, hidden_sz, FORMAT_CLASS)
        # Title / Suffix classifier neural networks
        self.guide_title = NameFormatModel(PRINTABLE, hidden_sz=hidden_sz, output_sz=len(TITLE))
        self.guide_suffix = NameFormatModel(PRINTABLE, hidden_sz=hidden_sz, output_sz=len(SUFFIX))
        # Hyperparameters
        self.peak_prob = peak_prob
        self.num_layers = num_layers
        self.hidden_sz = hidden_sz
        self.noise_prob = noise_prob

    def model(self, observations={"output": 0}):
        with torch.no_grad():
            # Sample format
            # aux_format_id = int(dist.Categorical(AUX_FORMAT_PROBS).sample().item())
            main_format_id = int(dist.Categorical(MAIN_FORMAT_PROBS).sample().item())

            # Sample title, first name, middle name, last name, and/or suffix
            firstname, middlename, lastname = None, None, None

            # if has_title(aux_format_id):
            #     title = TITLE[int(pyro.sample(TITLE_ADD, dist.Categorical(TITLE_PROBS)).item())]

            # if has_suffix(aux_format_id):
            #     suffix = SUFFIX[int(pyro.sample(SUFFIX_ADD, dist.Categorical(SUFFIX_PROBS)).item())]

            # first & last name generation
            firstname = sample_name(self.model_fn, FIRST_NAME_ADD)
            lastname = sample_name(self.model_ln, LAST_NAME_ADD)

            # Middle Name generation
            has_middle = has_middle_name(main_format_id)
            middle_name_format_id = None
            if has_middle:
                middle_name_format_id = int(dist.Categorical(MIDDLE_NAME_FORMAT_PROBS).sample().item())
                middlenames = []

                for i in range(num_middle_name(middle_name_format_id)):
                    if has_middle_initial(middle_name_format_id):
                        initial_probs = torch.zeros(self.num_output_chars).to(DEVICE)
                        initial_probs[26:52] = 1 / 26
                        letter_idx = int(
                            pyro.sample(f"{MIDDLE_NAME_ADD}_{i}_0", dist.Categorical(initial_probs)).item())
                        initial_format_probs = torch.zeros(self.num_output_chars).to(DEVICE)
                        initial_format_probs[self.output_chars.index(EOS)] = 1.
                        pyro.sample(f"{MIDDLE_NAME_ADD}_{i}_1", dist.Categorical(initial_format_probs))
                        middlename = self.output_chars[letter_idx]  # For capital names
                    else:
                        middlename = sample_name(self.model_fn, f"{MIDDLE_NAME_ADD}_{i}")
                    middlenames.append(middlename)

            only_printables = [char for char in string.printable]
            noised_first = noise_name(firstname, only_printables, len(firstname) + 2)
            noised_middles = []
            if has_middle:
                for i in range(len(middlenames)):
                    middle_i = middlenames[i]
                    noised_middles.append(noise_name(middle_i, only_printables, len(middle_i) + 2))
            noised_last = noise_name(lastname, only_printables, len(lastname) + 2)

            observation_probs, character_format_probs, full_name = generate_obs_and_chars_probs(noised_first,
                                                                                                noised_middles,
                                                                                                noised_last,
                                                                                                main_format_id,
                                                                                                middle_name_format_id,
                                                                                                self.peak_prob)

            for i in range(len(observation_probs)):
                curr_format = pyro.sample(f"{CHAR_FORMAT_ADD}_{i}",
                                          dist.Categorical(torch.tensor(character_format_probs[i]).to(DEVICE)))

            pyro.sample("output", dist.Categorical(torch.tensor(observation_probs[:len(observation_probs)]).to(DEVICE)),
                        obs=observations["output"])

        parse = {'firstname': firstname, 'middlename': middlename, 'lastname': lastname}

        print(
            "first name: {}, middle name: {}, last name: {}, middle name format: {}, main format: {}".format(
                firstname, middlename, lastname, middle_name_format_id, main_format_id))

        return full_name, parse

    def guide(self, observations=None):
        X = observations['output']

        # Infer formats and parse
        pyro.module("format_forward_lstm", self.guide_format.forward_lstm)
        pyro.module("format_backward_lstm", self.guide_format.backward_lstm)
        pyro.module("format_fc1", self.guide_format.fc1)
        char_class_samples = self.guide_format.forward(X, CHAR_FORMAT_ADD)

        _, first, middles, last, _ = parse_name(X, char_class_samples)

        # if len(title) > 0:
        #     pyro.module("title", self.guide_title)
        #     title_tensor = name_to_idx_tensor(title[0], PRINTABLE)
        #     sample = classify_using_format_model(self.guide_title, title_tensor,
        #                                          torch.IntTensor([len(title_tensor)]).to(DEVICE), TITLE_ADD)
        #     title = TITLE[sample]

        # if len(suffix) > 0:
        #     pyro.module("suffix", self.guide_suffix)
        #     suffix_tensor = name_to_idx_tensor(suffix[0], PRINTABLE)
        #     sample = classify_using_format_model(self.guide_suffix, suffix_tensor,
        #                                          torch.IntTensor([len(suffix_tensor)]).to(DEVICE), SUFFIX_ADD)
        #     suffix = SUFFIX[sample]

        pyro.module("first_name", self.guide_fn)
        if len(first) > 0:
            input = name_to_idx_tensor(first[0], self.guide_fn.input)
            samples = self.guide_fn.forward(input, FIRST_NAME_ADD)
            first = ''.join(self.output_chars[s] for s in samples)

        middle_names = []
        if len(middles) > 0:
            for i in range(len(middles)):
                input = name_to_idx_tensor(middles[i], self.guide_fn.input)
                samples = self.guide_fn.forward(input, f"{MIDDLE_NAME_ADD}_{i}")
                middle_names.append(''.join(self.output_chars[s] for s in samples))

        if len(last) > 0:
            pyro.module("last_name", self.guide_ln)
            input = name_to_idx_tensor(last[0], self.guide_ln.input)
            samples = self.guide_ln.forward(input, LAST_NAME_ADD)
            last = ''.join(self.output_chars[s] for s in samples)

        # TODO!!! Have to add full name reconstruction

        return {'firstname': first, 'middlename': middle_names, 'lastname': last}

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
        if len(name_string) > MAX_STRING_LEN: raise Exception(f"Name string length cannot exceed {MAX_STRING_LEN}.")
        name_as_list = [c for c in name_string]
        return name_to_idx_tensor(name_as_list, PRINTABLE, max_length=True)

    def load_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        name_fp = os.path.join(folder, "name_" + filename)
        format_fp = os.path.join(folder, "format_" + filename)
        title_suffix_fp = os.path.join(folder, "title_suffix_" + filename)
        if not os.path.exists(name_fp) or not os.path.exists(format_fp) or not os.path.exists(noise_fp):
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
        }
        torch.save(name_content, name_fp)
        torch.save(format_content, format_fp)
        torch.save(title_suffix_content, title_suffix_fp)

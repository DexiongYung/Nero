import argparse
import pyro
import distance
import statistics
import pandas as pd
from utilities.noiser import *
from utilities.config import *

from const import *
from infcomp import NameParser, TITLE, SUFFIX
from score import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='filepath to config json', type=str, default='config/UNNAMED_SESSION.json')
parser.add_argument('--true_posterior', help='whether to sample from p(z|x) or q(z|x)', nargs='?', default=False,
                    type=bool)
parser.add_argument('--num_particles', help='# of particles to use for SIS', nargs='?', default=10, type=int)
parser.add_argument('--num_samples', help='# samples', nargs='?', default=3, type=int)
parser.add_argument('--noised', help='whether to noise the observed name', nargs='?', default=False, type=bool)
parser.add_argument('--test_set', help='path of the test set', nargs='?', default='data/test.csv')

args = parser.parse_args()

config = load_json(args.config)


name_parser = NameParser(config['rnn_num_layers'], config['rnn_hidden_size'], config['format_hidden_size'])
name_parser.load_checkpoint(filename=f"{config['session_name']}.pth.tar")

fn_correct_count = 0
mn_correct_count = 0
ln_correct_count = 0
fn_distances = []
mn_distances = []
ln_distances = []

test_data = pd.read_csv(args.test_set, keep_default_na=False)[:100]


def parse_to_append(result):
    if type(result) == str:
        to_append = result
    elif len(result) == 0:
        to_append = ''
    else:
        to_append = result[0]
    return to_append

def infer(observed_name):
    if args.true_posterior:
        traces = get_importance_traces(observed_name, name_parser, args.num_samples, args.num_particles)
    else:
        traces = get_guide_traces(observed_name, name_parser, args.num_samples)
    log_probs = torch.tensor(list(map(lambda trace: trace.log_prob_sum().item(), traces)))
    max_prob_trace = traces[torch.argmax(log_probs, dim=-1).item()]
    parse = get_full_result(max_prob_trace, name_parser)
    firstname, middlename, lastname = parse['firstname'], parse['middlename'], parse['lastname']
    return firstname, middlename, lastname


for i, j in test_data.iterrows():

    curr = j['name']
    if args.noised:
        allowed_noise = [c for c in string.ascii_letters + string.digits]
        curr = noise_name(curr,allowed_noise)
    correct_fn = j['first']
    correct_mn = j['middle']
    correct_ln = j['last']

    fn_results = []
    mn_results = []
    ln_results = []

    firstname, middlename, lastname = infer(curr)

    fn_results.append(firstname)
    mn_results.append(middlename)
    ln_results.append(lastname)

    fn_mode = max(set(fn_results), key=fn_results.count)
    mn_mode = max(set(mn_results), key=mn_results.count)
    ln_mode = max(set(ln_results), key=ln_results.count)

    fn_distance = distance.levenshtein(fn_mode, correct_fn)
    mn_distance = distance.levenshtein(mn_mode, correct_mn)
    ln_distance = distance.levenshtein(ln_mode, correct_ln)
    fn_correct_count = fn_correct_count + 1 if fn_distance==0 else fn_correct_count
    mn_correct_count = mn_correct_count + 1 if mn_distance == 0 else mn_correct_count
    ln_correct_count = ln_correct_count + 1 if ln_distance == 0 else ln_correct_count
    fn_distances.append(fn_distance)
    mn_distances.append(mn_distance)
    ln_distances.append(ln_distance)

fn_average_distance = statistics.mean(fn_distances)
mn_average_distance = statistics.mean(mn_distances)
ln_average_distance = statistics.mean(ln_distances)
print("First name average number of letters wrong: %.3f"% fn_average_distance)
print("Middle name average number of letters wrong: %.3f"% mn_average_distance)
print("Last name average number of letters wrong: %.3f"% ln_average_distance)

fn_accuracy_rate = fn_correct_count/len(fn_distances)
mn_accuracy_rate = mn_correct_count/len(mn_distances)
ln_accuracy_rate = ln_correct_count/len(ln_distances)
print("First name accuracy: %.3f" % fn_accuracy_rate)
print("Middle name accuracy: %.3f" % mn_accuracy_rate)
print("Last name accuracy: %.3f" % ln_accuracy_rate)

fn_variance = statistics.variance(fn_distances)
mn_variance = statistics.variance(mn_distances)
ln_variance = statistics.variance(ln_distances)
print("First name variance: %.3f"% fn_variance)
print("Middle name variance: %.3f"% mn_variance)
print("Last name variance: %.3f"% ln_variance)
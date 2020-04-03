"""
1. Return just the parse based on format
2. Return probabilities associated with each trace in q
3. Return probabilities associated with each trace in p
Compute probability of each sample from q and take the max one
"""
from utilities.infcomp_utilities import FORMAT_CLASS
import argparse
import pyro

from const import *
from infcomp import NameParser, TITLE, SUFFIX
from utilities.config import load_json


def get_parse_result(sample_trace) -> dict:
    rv_names = sample_trace.stochastic_nodes
    title, first, middle, last, suffix = [], [], [], [], []
    for i in range(MAX_STRING_LEN):
        format_class = FORMAT_CLASS[sample_trace.nodes[f"char_format_{i}"]['value'].item()]
        if format_class == 't':
            title.append(i)
        elif format_class == 'f':
            first.append(i)
        elif format_class == 'm':
            middle.append(i)
        elif format_class == 'l':
            last.append(i)
        elif format_class == 's':
            suffix.append(i)
    
    prev_index = middle[0]-1 if len(middle)>0 else -1
    discontinuity_index = -1
    for i in middle:
        if i == prev_index+1:
            prev_index = i
        else:
            discontinuity_index = i
    
    def index_to_component(name_index_tensor, indexes) -> str:
        component = ''
        for i in indexes: 
            toadd = PRINTABLE[name_index_tensor[i]]
            if toadd != PAD and toadd != EOS: component += toadd
        return component
    
    name = sample_trace.nodes['_INPUT']['kwargs']['observations']['output']
    if discontinuity_index > 0:
        middle_component = index_to_component(name, middle[:discontinuity_index]) + ' ' + index_to_component(name, middle[discontinuity_index:])
    else:
        middle_component = index_to_component(name, middle)
    
    return {
        'title': index_to_component(name, title),
        'firstname': index_to_component(name, first),
        'middlename': middle_component,
        'lastname': index_to_component(name, last),
        'suffix': index_to_component(name, suffix),
    }

def get_full_result(sample_trace, name_parser) -> dict:
    rv_names = sample_trace.stochastic_nodes

    title = ''
    firstname = ''
    middlename = ''
    middlename1 = ''
    middlename2 = ''
    lastname = ''
    suffix = ''

    if "title" in rv_names:
        title = TITLE[sample_trace.nodes[f"title"]['value'].item()]
    if "suffix" in rv_names:
        suffix = SUFFIX[sample_trace.nodes[f"suffix"]['value'].item()]
    for i in range(MAX_OUTPUT_LEN):
        if f"first_name_{i}" in rv_names:
            if name_parser.output_chars[sample_trace.nodes[f"first_name_{i}"]['value'].item()] != EOS:
                firstname += name_parser.output_chars[sample_trace.nodes[f"first_name_{i}"]['value'].item()]
        if f"middle_name_0_{i}" in rv_names:
            if name_parser.output_chars[sample_trace.nodes[f"middle_name_0_{i}"]['value'].item()] != EOS:
                middlename1 += name_parser.output_chars[sample_trace.nodes[f"middle_name_0_{i}"]['value'].item()]
        if f"middle_name_1_{i}" in rv_names:
            if name_parser.output_chars[sample_trace.nodes[f"middle_name_1_{i}"]['value'].item()] != EOS:
                middlename2 += name_parser.output_chars[sample_trace.nodes[f"middle_name_1_{i}"]['value'].item()]
        if f"last_name_{i}" in rv_names:
            if name_parser.output_chars[sample_trace.nodes[f"last_name_{i}"]['value'].item()] != EOS:
                lastname += name_parser.output_chars[sample_trace.nodes[f"last_name_{i}"]['value'].item()]

    if middlename1 != '' and middlename2 != '':
        middlename = middlename1 + ' ' + middlename2
    if middlename1 != '':
        middlename = middlename1

    return {
        'title': title, 
        'firstname': firstname, 
        'middlename': middlename, 
        'lastname': lastname,
        'suffix': suffix
    }

def get_importance_traces(name, name_parser, num_samples, num_particles) -> list:
    sample_traces = []
    csis = pyro.infer.CSIS(name_parser.model, name_parser.guide, pyro.optim.Adam({'lr': 0.001}),
                            num_inference_samples=num_particles)
    posterior = csis.run(observations={'output': name_parser.get_observes(name)})
    particle_weights = csis.get_normalized_weights()
    # print(f"Importance Weights: {particle_weights}")
    for _ in range(num_samples):
        sample_traces.append(posterior())
    return sample_traces

def get_guide_traces(name, name_parser, num_samples) -> list:
    sample_traces = []
    for _ in range(num_samples):
        trace = pyro.poutine.trace(name_parser.guide).get_trace(observations={'output': name_parser.get_observes(name)})
        sample_traces.append(trace)
    return sample_traces



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='filepath to config json', type=str, default='config/3rnn.json')
    parser.add_argument('--name', help='name to parse', nargs='?', default='Dr. Wood, Frnnk Donald', type=str)
    parser.add_argument('--true_posterior', help='whether to sample from p(z|x) or q(z|x)', nargs='?', default=True,
                        type=bool)
    parser.add_argument('--num_particles', help='# of particles to use for SIS', nargs='?', default=15, type=int)
    parser.add_argument('--num_samples', help='# samples', nargs='?', default=10, type=int)
    args = parser.parse_args()

    config = load_json(args.config)
    name_parser = NameParser(config['rnn_num_layers'], config['rnn_hidden_size'])
    name_parser.load_checkpoint(filename=f"{config['session_name']}.pth.tar")

    if args.true_posterior:
        sample_traces = get_importance_traces(args.name, name_parser, args.num_samples, args.num_particles)
    else:
        sample_traces = get_guide_traces(args.name, name_parser, args.num_samples)

    for j, sample in enumerate(sample_traces):
        #print("Trace Probability: %.5f" % sample.log_prob_sum().exp())
        #print(f"Parsed Result: {get_parse_result(sample)}")
        print(f"Trace Result:  {get_full_result(sample)}")


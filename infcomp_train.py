import argparse
import matplotlib.pyplot as plt
import pyro
from pyro.infer import CSIS
from infcomp import NameParser
from utilities.config import save_json

pyro.enable_validation(True)

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session',
                    nargs='?', default='prob_prog', type=str)
parser.add_argument('--rnn_hidden_size',
                    help='Size of RNN hidden layers', nargs='?', default=512, type=int)
parser.add_argument('--format_hidden_size',
                    help='Hidden size for format models', nargs='?', default=256, type=int)
parser.add_argument('--rnn_num_layers',
                    help='Number of RNNs to stack', nargs='?', default=6, type=int)
parser.add_argument('--char_error_rate', help="Probability of randomly permuting a single character in the likelihood",
                    nargs='?', default=0., type=float)
parser.add_argument('--lr', help='Learning rate',
                    nargs='?', default=0.001, type=float)
parser.add_argument('--num_particles', help='Number of particles to evaluate for loss',
                    nargs='?', default=10, type=int)
parser.add_argument('--num_steps', help='Number of gradient descent steps',
                    nargs='?', default=50000, type=int)
parser.add_argument('--continue_training',
                    help='An int deciding whether to keep training the model with config name, 0=False, 1=True',
                    nargs='?',
                    default=True, type=bool)

# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
SESSION_NAME = args.name
to_save = {
    'session_name': args.name,
    'format_hidden_size': args.format_hidden_size,
    'rnn_hidden_size': args.rnn_hidden_size,
    'rnn_num_layers': args.rnn_num_layers,
    'char_error_rate': args.char_error_rate,
    'lr': args.lr,
    'num_steps': args.num_steps,
    'num_particles': args.num_particles
}

save_json(f'config/{SESSION_NAME}.json', to_save)

name_parser = NameParser(num_layers=args.rnn_num_layers, hidden_sz=args.rnn_hidden_size,
                         peak_prob=1. - args.char_error_rate, format_hidden_sz=args.format_hidden_size)
optimizer = pyro.optim.Adam({'lr': args.lr})

if args.continue_training:
    name_parser.load_checkpoint(filename=f"{SESSION_NAME}.pth.tar")

csis = pyro.infer.CSIS(name_parser.model, name_parser.guide,
                       optimizer, training_batch_size=args.num_particles)

losses = []
for step in range(args.num_steps):
    try:
        loss = csis.step()
        if step % 1 == 0:
            print(f"step: {step} - loss: {loss}")
            losses.append(loss)
        if step > 0 and step % 10 == 0:
            print(f"Saving plot to result/{SESSION_NAME}.png...")
            plt.plot(losses)
            plt.title("Infcomp Loss")
            plt.xlabel("steps")
            plt.ylabel("loss")
            plt.savefig(f"result/{SESSION_NAME}.png")
            name_parser.save_checkpoint(filename=f"{SESSION_NAME}.pth.tar")
    except RuntimeError:
        continue

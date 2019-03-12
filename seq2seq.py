# Python imports.
import pickle
import argparse
import numpy as np
import pdb

# Other imports.
from utils import create_data_splits, get_lines
from train import train
from evaluate import evaluate
from tensorboardX import SummaryWriter
from hyperparameters import Hyperparameters
from vocab import Vocab


def main():
    french = get_lines(args.french_filename)
    english = get_lines(args.english_filename)

    french_english_pairs = list(zip(french, english))

    training_pairs, validation_pairs = create_data_splits(french_english_pairs)

    training_input_sentences = [pair[0] for pair in training_pairs]
    training_output_sentences = [pair[1] for pair in training_pairs]
    validation_input_sentences = [pair[0] for pair in validation_pairs]
    validation_output_sentences = [pair[1] for pair in validation_pairs]

    Vocab(args.english_filename, args.french_filename)

    with open("data/bpe_vocab.pkl", "rb") as f:
        v = pickle.load(f)
    with open("data/bpe_reverse_vocab.pkl", "rb") as f:
        rv = pickle.load(f)

    t_loss, t_accuracy = train(training_input_sentences, training_output_sentences, v, rv, hyperparameters, writer)
    print("Training Accuracy = {:.1f}".format(100. * np.mean(t_accuracy)))

    evaluate(validation_input_sentences, validation_output_sentences, v, rv, hyperparameters, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--french_filename", type=str, help="Path to french corpus")
    parser.add_argument("--english_filename", type=str, help="Path to english corpus")
    parser.add_argument("--experiment_name", type=str, help="Name of experiment", default="")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--epochs", type=int, help="number of training epochs", default=1)
    parser.add_argument("--embedding_size", type=int, help="embedding size", default=512)
    parser.add_argument("--hidden_size", type=int, help="RNN size", default=256)
    parser.add_argument("--lr", type=float, help="Learning rate", default=2e-3)
    parser.add_argument("--bidirectional", type=bool, help="Bidirectional RNN", default=True)
    parser.add_argument("--num_rnn_layers", type=int, help="# RNN Layers", default=1)
    args = parser.parse_args()

    writer = SummaryWriter(args.experiment_name)
    hyperparameters = Hyperparameters(args)

    main()

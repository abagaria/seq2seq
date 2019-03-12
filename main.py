# Python imports.
import pickle
import argparse
import numpy as np
import pdb
import os

# Other imports.
from utils import create_data_splits, get_lines

from tensorboardX import SummaryWriter
from hyperparameters import Hyperparameters


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            print("Unable to create {}".format(dir_path))


def create_vocab(french_file, english_file, version):
    if version == "vanilla":
        from vanilla.vocab import Vocab
        Vocab(french_file)
        Vocab(english_file)
    else:
        from bpe.vocab import Vocab
        Vocab(french_file, english_file)


def load_vocab(french_file, english_file, version):
    if version == "bpe":
        with open("data/bpe_vocab.pkl", "rb") as f:
            v = pickle.load(f)
        with open("data/bpe_reverse_vocab.pkl", "rb") as f:
            rv = pickle.load(f)
        return v, rv

    # Vanilla
    path_to_english_vocab = english_file.split(".")[0] + "_vocab.pkl"
    path_to_french_vocab  = french_file.split(".")[0] + "_vocab.pkl"
    path_to_english_reverse = english_file.split(".")[0] + "_reverse_vocab.pkl"
    path_to_french_reverse = french_file.split(".")[0] + "_reverse_vocab.pkl"

    with open(path_to_french_vocab, "rb") as f:
        fv = pickle.load(f)
    with open(path_to_french_reverse, "rb") as f:
        frv = pickle.load(f)
    with open(path_to_english_vocab, "rb") as f:
        ev = pickle.load(f)
    with open(path_to_english_reverse, "rb") as f:
        erv = pickle.load(f)

    return fv, frv, ev, erv


def main():
    create_directory("{}/saved_runs".format(args.version))
    
    french = get_lines(args.french_file)
    english = get_lines(args.english_file)

    create_vocab(args.french_file, args.english_file, args.version)

    french_english_pairs = list(zip(french, english))
    training_pairs, validation_pairs = create_data_splits(french_english_pairs)

    training_input_sentences = [pair[0] for pair in training_pairs]
    training_output_sentences = [pair[1] for pair in training_pairs]
    validation_input_sentences = [pair[0] for pair in validation_pairs]
    validation_output_sentences = [pair[1] for pair in validation_pairs]

    if args.version == "bpe":
        from bpe.train import train
        from bpe.evaluate import evaluate

        v, rv = load_vocab(args.french_file, args.english_file, args.version)
        training_perplexity = train(training_input_sentences, training_output_sentences, v, rv, hyperparameters, writer)
        print("training perplexity = {}".format(training_perplexity))
        evaluate(validation_input_sentences, validation_output_sentences, v, rv, hyperparameters, writer)

    else:
        from vanilla.train import train
        from vanilla.evaluate import evaluate

        fv, frv, ev, erv = load_vocab(args.french_file, args.english_file, args.version)
        training_perplexity = train(training_input_sentences, training_output_sentences, fv, ev, frv, erv, hyperparameters, writer)
        print("training perplexity = {}".format(training_perplexity))
        evaluate(validation_input_sentences, validation_output_sentences, fv, ev, frv, erv, hyperparameters, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="vanilla/bpe")
    parser.add_argument("--english_file", help="path to english file")
    parser.add_argument("--french_file", help="path to french file")
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--epochs", type=int, help="number of training epochs", default=10)
    parser.add_argument("--embedding_size", type=int, help="embedding size", default=500)
    parser.add_argument("--hidden_size", type=int, help="RNN size", default=512)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--bidirectional", type=bool, help="Bidirectional RNN", default=False)
    parser.add_argument("--num_rnn_layers", type=int, help="# RNN Layers", default=1)
    args = parser.parse_args()

    if not args.version in ["vanilla", "bpe"]:
        raise ValueError("{} should be vanilla or bpe".format(args.version))

    if not os.path.exists(args.english_file):
        raise ValueError("{} does not exist, call preprocess on the corpus before".format(args.english_file))
    if not os.path.exists(args.french_file):
        raise ValueError("{} does not exist, call preprocess on the corpus before".format(args.english_file))

    writer = SummaryWriter()
    hyperparameters = Hyperparameters(args)

    main()

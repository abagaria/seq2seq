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


def main():
    french = get_lines("data/french-toy.txt")
    english = get_lines("data/english-toy.txt")

    french_english_pairs = list(zip(french, english))

    training_pairs, validation_pairs = create_data_splits(french_english_pairs)

    training_input_sentences = [pair[0] for pair in training_pairs]
    training_output_sentences = [pair[1] for pair in training_pairs]
    validation_input_sentences = [pair[0] for pair in validation_pairs]
    validation_output_sentences = [pair[1] for pair in validation_pairs]

    with open("data/french_vocab.pkl", "rb") as f:
        fv = pickle.load(f)
    with open("data/french_reverse_vocab.pkl", "rb") as f:
        frv = pickle.load(f)
    with open("data/english_vocab.pkl", "rb") as f:
        ev = pickle.load(f)
    with open("data/english_reverse_vocab.pkl", "rb") as f:
        erv = pickle.load(f)

    t_loss, t_accuracy = train(training_input_sentences, training_output_sentences, fv, ev, frv, erv, hyperparameters, writer)
    print("Training Accuracy = {:.1f} +/ {:.1f}".format(100. * np.mean(t_accuracy), 100. * np.std(t_accuracy)))

    evaluate(validation_input_sentences, validation_output_sentences, fv, ev, frv, erv, hyperparameters, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Name of experiment", default="")
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--epochs", type=int, help="number of training epochs", default=10)
    parser.add_argument("--embedding_size", type=int, help="embedding size", default=500)
    parser.add_argument("--hidden_size", type=int, help="RNN size", default=512)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--bidirectional", type=bool, help="Bidirectional RNN", default=False)
    parser.add_argument("--num_rnn_layers", type=int, help="# RNN Layers", default=2)
    args = parser.parse_args()

    writer = SummaryWriter(args.experiment_name)
    hyperparameters = Hyperparameters(args)

    main()

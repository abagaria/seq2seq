# Python imports.
import pickle
import argparse
import numpy as np
import pdb
import os

# Other imports.
from utils import create_data_splits, get_lines

# from tensorboardX import SummaryWriter
from hyperparameters import Hyperparameters


def split(file, flip):
    input_lines = []
    output_lines = []
    with open(file, "r") as f:
        for line in f:
            if not flip:
                input_line = line.split("\t")[0]
                output_line = line.split("\t")[1][:-1]
            else:
                input_line = line.split("\t")[1][:-1]
                output_line = line.split("\t")[0]
            input_lines.append(input_line)
            output_lines.append(output_line)
    return input_lines, output_lines


def attach_target_token(input_lines, target_token="<2e>"):
    new_lines = []
    for line in input_lines:
        new_line = target_token + " " + line
        new_lines.append(new_line)
    return new_lines


def create_input_output_file(e2g_train_file, e2f_train_file):
    combined_file = e2g_train_file.split(".")[0] + e2f_train_file.split("/")[-1]
    print("creating combined file: ", combined_file)
    german_lines, english_lines_1 = split(e2g_train_file, flip=True)
    english_lines_2, french_lines = split(e2f_train_file, flip=False)
    german_lines = attach_target_token(german_lines, "<2e>")
    english_lines_2 = attach_target_token(english_lines_2, "<2f>")
    with open(combined_file, "w+") as f:
        for gl, el1, fl, el2 in zip(german_lines, english_lines_1, french_lines, english_lines_2):
            f.write(gl + "\t" + el1 + "\n")
            f.write(el2 + "\t" + fl + "\n")
    input_file = combined_file.split(".")[0] + "_input_lines.txt"
    output_file = combined_file.split(".")[0] + "_output_lines.txt"
    with open(input_file, "w+") as wf:
        with open(combined_file, "r") as rf:
            for line in rf:
                input_line = line.split("\t")[0] + "\n"
                wf.write(input_line)
    with open(output_file, "w+") as wf:
        with open(combined_file, "r") as rf:
            for line in rf:
                output_line = line.split("\t")[1]
                wf.write(output_line)

    return input_file, output_file


def prepare_test_file(e2g_test_file, e2f_test_file):
    g_lines, f_lines = [], []
    out_file = e2f_test_file.split(".")[0] + e2f_test_file.split("/")[-1]
    with open(e2g_test_file, "r") as e2gf:
        with open(e2f_test_file, "r") as e2ff:
            for e2g, e2f in zip(e2gf, e2ff):
                g_lines.append(e2g.split("\t")[1][:-1])
                f_lines.append(e2f.split("\t")[1][:-1])
    with open(out_file, "w+") as f:
        for g, fl in zip(g_lines, f_lines):
            f.write(g + "\t" + fl + "\n")
    return g_lines, f_lines


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            print("Unable to create {}".format(dir_path))


def create_vocab(input_file, output_file):
    from bpe.vocab import Vocab
    Vocab(input_file, output_file)


def load_vocab():
    with open("data/bpe_vocab.pkl", "rb") as f:
        v = pickle.load(f)
    with open("data/bpe_reverse_vocab.pkl", "rb") as f:
        rv = pickle.load(f)
    return v, rv


def main():
    create_directory("saved_runs")

    input_lines = get_lines(_input_file)
    output_lines = get_lines(_output_file)

    create_vocab(_input_file, _output_file)

    training_pairs = list(zip(input_lines, output_lines))

    training_input_sentences = [pair[0] for pair in training_pairs]
    training_output_sentences = [pair[1] for pair in training_pairs]
    validation_input_sentences, validation_output_sentences = prepare_test_file(_val_input_file, _val_output_file)

    from bpe.train import train
    from bpe.evaluate import evaluate

    v, rv = load_vocab()
    training_perplexity = train(training_input_sentences, training_output_sentences, v, rv, hyperparameters, writer)
    print("training perplexity = {}".format(training_perplexity))
    evaluate(validation_input_sentences, validation_output_sentences, v, rv, hyperparameters, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2g_train_file", type=str, default="data/oneShotDEFR/dattrainED.txt")
    parser.add_argument("--e2g_test_file", type=str, default="data/oneShotDEFR/dattestED.txt")
    parser.add_argument("--e2f_train_file", type=str, default="data/oneShotDEFR/dattrainEF.txt")
    parser.add_argument("--e2f_test_file", type=str, default="data/oneShotDEFR/dattestEF.txt")
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--epochs", type=int, help="number of training epochs", default=10)
    parser.add_argument("--embedding_size", type=int, help="embedding size", default=500)
    parser.add_argument("--hidden_size", type=int, help="RNN size", default=512)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--bidirectional", type=bool, help="Bidirectional RNN", default=False)
    parser.add_argument("--num_rnn_layers", type=int, help="# RNN Layers", default=1)
    args = parser.parse_args()

    _input_file, _output_file = create_input_output_file(args.e2g_train_file, args.e2f_train_file)
    _val_input_file, _val_output_file = args.e2g_test_file, args.e2f_test_file

    # writer = SummaryWriter()
    writer = None

    hyperparameters = Hyperparameters(args)

    main()

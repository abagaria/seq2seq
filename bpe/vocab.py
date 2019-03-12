# Python imports.
import pickle
from collections import defaultdict


class Vocab(object):
    """ Class used to create a vocabulary over the english (natural) language sentences in the corpus. """

    START_TOKEN = "<SOS>"
    STOP_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"

    def __init__(self, input_file_1, input_file_2):
        self.input_file_1 = input_file_1
        self.input_file_2 = input_file_2
        self.training_data = self._get_training_data()
        self.vocab, self.reverse_vocab = self._create_vocab()
        self._save_vocab()

    def _get_training_data(self):
        train_data = []
        with open(self.input_file_1) as _file:
            for line in _file:
                train_data.append(line)
        with open(self.input_file_2) as _file:
            for line in _file:
                train_data.append(line)
        return train_data

    def _create_vocab(self):
        all_words = []
        for line in self.training_data:
            words = line.split()
            all_words += words
        word_set = set(all_words)
        vocab = defaultdict()
        reverse_vocab = defaultdict()

        final_idx = 0
        for idx, word in enumerate(word_set):
            vocab[word] = idx + 1
            reverse_vocab[idx + 1] = word
            final_idx = idx + 1

        vocab[self.PAD_TOKEN] = 0
        vocab[self.START_TOKEN] = final_idx + 1
        vocab[self.STOP_TOKEN] = final_idx + 2

        reverse_vocab[0] = self.PAD_TOKEN
        reverse_vocab[final_idx + 1] = self.START_TOKEN
        reverse_vocab[final_idx + 2] = self.STOP_TOKEN

        return vocab, reverse_vocab

    def _save_vocab(self):
        vocab_filename = "data/bpe_vocab.pkl"
        rev_vocab_filename = "data/bpe_reverse_vocab.pkl"
        with open(vocab_filename, "wb") as vf:
            pickle.dump(self.vocab, vf)
        with open(rev_vocab_filename, "wb") as ivf:
            pickle.dump(self.reverse_vocab, ivf)


if __name__ == "__main__":
    in_file_1 = "data/english-bpe.txt"
    in_file_2 = "data/french-bpe.txt"
    v = Vocab(in_file_1, in_file_2)

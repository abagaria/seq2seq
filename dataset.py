# Python imports.
import pickle
import pdb
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class NMTDataset(Dataset):
    """ Dataset module for Neural Machine Translation . """

    def __init__(self, french_sentences, english_sentences, french_vocab, english_vocab,
                 french_reverse_vocab, english_reverse_vocab):
        """
        The task is to translate French sentences into English.
        Args:
            french_sentences (list)
            english_sentences (list)
            french_vocab (defaultdict)
            english_vocab (defaultdict)
            french_reverse_vocab (defaultdict)
            english_reverse_vocab (defaultdict)
        """
        self.input_sentences = french_sentences
        self.output_sentences = english_sentences

        self.input_vocab = french_vocab
        self.output_vocab = english_vocab

        self.input_reverse_vocab = french_reverse_vocab
        self.output_reverse_vocab = english_reverse_vocab

        self.input_unk_token = french_vocab["UNK"]
        self.output_unk_token = english_vocab["UNK"]

        self.input_tensors = self.read_input_sentences(french_sentences)
        self.output_tensors = self.read_output_sentences(english_sentences)

        # self.padded_input_tensors = self.pad_sentences(self.input_tensors)
        # self.padded_output_tensors = self.pad_sentences(self.output_tensors)

    @staticmethod
    def pad_sentences(sentence_tensors):
        max_length = max(map(len, sentence_tensors))
        padded_tensor = torch.zeros(len(sentence_tensors), max_length, dtype=torch.long)
        for i, sentence_tensor in enumerate(sentence_tensors):
            seq_length = sentence_tensor.shape[1]
            padded_tensor[i, :seq_length] = sentence_tensor
        return padded_tensor

    def read_input_sentences(self, sentences):
        sentence_tensors = []
        for sentence in sentences:
            sentence_tensors.append(self.read_input_sentence(sentence))
        return sentence_tensors

    def read_input_sentence(self, sentence):
        sequence = sentence.split()
        word_ids = [self.input_vocab[word] if word in self.input_vocab else self.input_unk_token for word in sequence]
        return torch.tensor(word_ids, dtype=torch.long)

    def read_output_sentences(self, sentences):
        sentence_tensors = []
        for sentence in sentences:
            sentence_tensors.append(self.read_output_sentence(sentence))
        return sentence_tensors

    def read_output_sentence(self, sentence):
        sequence = sentence.split()
        word_ids = [self.output_vocab[word] if word in self.output_vocab else self.output_unk_token for word in sequence]
        return torch.tensor(word_ids, dtype=torch.long)

    def decode_english_line(self, word_ids):
        english_sentence = []
        for word_id in word_ids.tolist():
            english_sentence.append(self.output_reverse_vocab[word_id])
        return english_sentence

    def decode_french_line(self, word_ids):
        french_sentence = []
        for word_id in word_ids.tolist():
            french_sentence.append(self.input_reverse_vocab[word_id])
        return french_sentence

    def __len__(self):
        assert len(self.input_tensors) == len(self.output_tensors), "Each sentence should have 1 label"
        return len(self.input_tensors)

    # TODO: 1st write version for batch size 1 and then extend to batch size N
    def __getitem__(self, i):
        input_tensor = self.input_tensors[i]
        output_tensor = self.output_tensors[i]

        original_input_length = input_tensor.shape[0]
        original_output_length = output_tensor.shape[0]

        # Add a <STOP> token to the end of the encoder input
        encoder_input = torch.ones(original_input_length + 1, dtype=torch.long)
        encoder_input[:original_input_length] = input_tensor
        encoder_input[original_input_length] = self.input_vocab["<EOS>"]

        # Add a <START> at the beginning of the decoder input
        decoder_input = torch.ones(original_output_length + 1, dtype=torch.long)
        decoder_input[0] = self.output_vocab["<SOS>"]
        decoder_input[1:] = output_tensor

        # Add a <STOP> token at the end of the decoder output
        decoder_output = torch.ones(original_output_length + 1, dtype=torch.long)
        decoder_output[:original_output_length] = output_tensor
        decoder_output[original_output_length] = self.output_vocab["<EOS>"]

        return encoder_input, decoder_input, decoder_output


def extract_sentences(_file):
    sentences = []
    with open(_file) as _f:
        for line in _f:
            sentences.append(line)
    return sentences


if __name__ == "__main__":
    french = extract_sentences("data/french.txt")
    english = extract_sentences("data/english.txt")
    with open("data/french_vocab.pkl", "rb") as f:
        fv = pickle.load(f)
    with open("data/french_reverse_vocab.pkl", "rb") as f:
        frv = pickle.load(f)
    with open("data/english_vocab.pkl", "rb") as f:
        ev = pickle.load(f)
    with open("data/english_reverse_vocab.pkl", "rb") as f:
        erv = pickle.load(f)
    d_set = NMTDataset(french, english, fv, ev, frv, erv)

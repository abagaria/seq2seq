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


def collate_fn(data):
    """
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Sequences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data (list): of tuples of the form <encoder_input, decoder_input, decoder_output>

    Returns:
        input_sequences (torch.tensor)
        input_lengths (list)
        output_sequences (torch.tensor)
        output_lengths (list)
    """
    def merge(sequences):
        lengths = list(map(len, sequences))
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, torch.tensor(lengths).long()

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    encoder_inputs, decoder_inputs, decoder_outputs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    padded_encoder_inputs, encoder_input_lengths = merge(encoder_inputs)
    padded_decoder_inputs, decoder_input_lengths = merge(decoder_inputs)
    padded_decoder_outputs, decoder_output_lengths = merge(decoder_outputs)

    return padded_encoder_inputs, encoder_input_lengths, padded_decoder_inputs, decoder_input_lengths,\
        padded_decoder_outputs, decoder_output_lengths


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

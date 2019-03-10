# Python imports.
import pdb

# PyTorch imports.
import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """ Encoder network. """

    def __init__(self, input_vocab_size, embedding_size, hidden_size, rnn_layers, bidirectional, device):
        super(EncoderRNN, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding(input_vocab_size, embedding_size)
        self.rnn_encoder = nn.GRU(embedding_size, hidden_size, num_layers=rnn_layers,
                                  bidirectional=bidirectional, batch_first=True)

        self.to(device)

    def forward(self, tokens):
        embeds = self.embedding(tokens)
        output, hidden = self.rnn_encoder(embeds)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_size, device=self.device)

# Python imports.
import pdb

# PyTorch imports.
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderRNN(nn.Module):
    """ Encoder network. """

    def __init__(self, embedding_size, hidden_size, rnn_layers, bidirectional, device, dataset):
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.device = device
        self.dataset = dataset  # TODO

        self.rnn_encoder = nn.GRU(embedding_size, hidden_size, num_layers=rnn_layers,
                                  bidirectional=bidirectional, batch_first=True)

        self.to(device)

    def forward(self, embedded_tokens, seq_lens):
        packed_input = pack_padded_sequence(embedded_tokens, seq_lens, batch_first=True)
        _, hidden = self.rnn_encoder(packed_input)

        return hidden

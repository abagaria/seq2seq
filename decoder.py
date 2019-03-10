# Python imports.
import pdb

# PyTorch imports.
import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """ Decoder network. """

    def __init__(self, output_vocab_size, embedding_size, hidden_size, rnn_layers, bidirectional, device):
        super(DecoderRNN, self).__init__()
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding(output_vocab_size, embedding_size)
        self.rnn_decoder = nn.GRU(embedding_size, hidden_size, num_layers=rnn_layers,
                                  bidirectional=bidirectional, batch_first=True)
        self.classifier = nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_vocab_size)

        self.to(device)

    def forward(self, tokens, encoder_hidden):
        embeds = self.embedding(tokens)
        output, hidden = self.rnn_decoder(embeds, encoder_hidden)
        logits = self.classifier(output)
        return logits

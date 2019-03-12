# Python imports.
import pdb

# PyTorch imports.
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from encoder import EncoderRNN
from decoder import DecoderRNN


class Seq2Seq(nn.Module):
    """ Seq2Seq Network. """

    def __init__(self, embedding_size, hidden_size, vocab_size, rnn_layers, bidirectional, device, dataset):
        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.device = device
        self.dataset = dataset  # TODO

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = EncoderRNN(embedding_size, hidden_size, rnn_layers, bidirectional, device, dataset)
        self.decoder = DecoderRNN(vocab_size, embedding_size, hidden_size, rnn_layers, bidirectional, device, dataset)

    def forward(self, encoder_tokens, decoder_tokens, encoder_seq_lens, decoder_seq_lens):
        encoder_embeds = self.embedding(encoder_tokens)
        encoder_hidden = self.encoder(encoder_embeds, encoder_seq_lens)

        decoder_embeds = self.embedding(decoder_tokens)
        logits = self.decoder(decoder_embeds, decoder_seq_lens, encoder_hidden)
        return logits

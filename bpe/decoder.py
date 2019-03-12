# Python imports.
import pdb

# PyTorch imports.
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DecoderRNN(nn.Module):
    """ Decoder network. """

    def __init__(self, output_vocab_size, embedding_size, hidden_size, rnn_layers, bidirectional, device, dataset):
        super(DecoderRNN, self).__init__()
        self.output_vocab_size = output_vocab_size
        self.hidden_size = 2 * hidden_size if bidirectional else hidden_size
        self.embedding_size = embedding_size
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.device = device
        self.dataset = dataset  # TODO

        self.embedding = nn.Embedding(output_vocab_size, embedding_size)
        self.rnn_decoder = nn.GRU(embedding_size, self.hidden_size, num_layers=rnn_layers, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, output_vocab_size)

        self.to(device)

    def forward(self, embedded_tokens, seq_lens, encoder_hidden):
        sorted_seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        sorted_seq_tensor = embedded_tokens[perm_idx]

        decoder_hidden = self._cat_directions(encoder_hidden) if self.bidirectional else encoder_hidden

        packed_input = pack_padded_sequence(sorted_seq_tensor, sorted_seq_lens, batch_first=True)
        packed_output, _ = self.rnn_decoder(packed_input, decoder_hidden)
        output, output_lens = pad_packed_sequence(packed_output, batch_first=True)

        logits = self.classifier(output)

        _, unsorted_idx = perm_idx.sort(0)
        unsorted_logits = logits[unsorted_idx].squeeze(1)

        return unsorted_logits

    def _cat_directions(self, h):
        return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

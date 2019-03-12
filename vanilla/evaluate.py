# Python imports.
from tqdm import tqdm
import numpy as np
import pdb

# PyTorch imports.
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Other imports.
from vanilla.dataset import NMTDataset
from vanilla.dataset import collate_fn
from vanilla.encoder import EncoderRNN
from vanilla.decoder import DecoderRNN


def evaluate(input_sentences, output_sentences, input_vocab, output_vocab, input_reverse, output_reverse, hy, writer):
    dataset = NMTDataset(input_sentences, output_sentences, input_vocab, output_vocab, input_reverse, output_reverse)
    loader = DataLoader(dataset, batch_size=hy.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_vocab_size = len(input_vocab.keys())
    output_vocab_size = len(output_vocab.keys())

    encoder = EncoderRNN(input_vocab_size, hy.embedding_size, hy.hidden_size, hy.rnn_layers, hy.bidirectional, device, dataset)
    decoder = DecoderRNN(output_vocab_size, hy.embedding_size, hy.hidden_size, hy.rnn_layers, hy.bidirectional, device, dataset)

    accuracies, perplexities = [], []

    for epoch in range(1, hy.num_epochs + 1):
        encoder.load_state_dict(torch.load("vanilla/saved_runs/encoder_{}_weights.pt".format(epoch)))
        decoder.load_state_dict(torch.load("vanilla/saved_runs/decoder_{}_weights.pt".format(epoch)))
        accuracy, perplexity = compute_model_accuracy(encoder, decoder, output_vocab, loader, device, epoch, writer)
        accuracies.append(accuracy)
        perplexities.append(perplexity)

    print("=" * 80)
    print("Evaluation metrics:")
    print("Final Accuracy = {:.1f}".format(100.*np.max(accuracies)))
    print("Final perplexity = {:.2f}".format(np.max(perplexities)))
    print("=" * 80)

    return accuracies


def compute_model_accuracy(encoder, decoder, vocab, loader, device, epoch, writer):
    num_correct = 0
    num_total = 0
    loss_history = []

    loss_function = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"]).to(device)

    encoder.eval()
    decoder.eval()

    print("\rComputing validation accuracy model @ {} epoch..".format(epoch))

    for encoder_input, encoder_len, decoder_input, decoder_input_len, decoder_output, decoder_output_len in loader:
        # Move the data to the GPU
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        decoder_output = decoder_output.to(device)
        encoder_len = encoder_len.to(device)
        decoder_input_len = decoder_input_len.to(device)

        with torch.no_grad():
            encoder_hidden = encoder(encoder_input, encoder_len)
            logits = decoder(decoder_input, decoder_input_len, encoder_hidden)
            loss = loss_function(logits.view(encoder_input.shape[0] * decoder_output.shape[1], -1), decoder_output.view(-1))
            predicted_sequence = logits.argmax(dim=-1)

        for i in range(encoder_input.shape[0]):
            output_length = decoder_output_len[i]
            predictions = predicted_sequence[i, :output_length]
            ground_truth = decoder_output[i, :output_length]

            num_correct += (predictions == ground_truth).sum().item()
            num_total += output_length

        loss_history.append(loss.item())

    accuracy = (1. * num_correct) / float(num_total)
    perplexity = np.exp(np.mean(loss_history))

    writer.add_scalar("validation_accuracy", accuracy, epoch)

    return accuracy, perplexity

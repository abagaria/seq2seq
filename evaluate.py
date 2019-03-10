# Python imports.
from tqdm import tqdm
import numpy as np

# PyTorch imports.
import torch
from torch.utils.data import DataLoader

# Other imports.
from dataset import NMTDataset
from hyperparameters import *
from encoder import EncoderRNN
from decoder import DecoderRNN


def evaluate(input_sentences, output_sentences, input_vocab, output_vocab, input_reverse, output_reverse, hy, writer):
    dataset = NMTDataset(input_sentences, output_sentences, input_vocab, output_vocab, input_reverse, output_reverse)
    loader = DataLoader(dataset, batch_size=hy.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_vocab_size = len(input_vocab.keys())
    output_vocab_size = len(output_vocab.keys())

    encoder = EncoderRNN(input_vocab_size, hy.embedding_size, hy.hidden_size, hy.rnn_layers, hy.bidirectional, device)
    decoder = DecoderRNN(output_vocab_size, hy.embedding_size, hy.hidden_size, hy.rnn_layers, hy.bidirectional, device)

    accuracies = []

    for epoch in range(1, hy.num_epochs + 1):
        encoder.load_state_dict(torch.load("saved_runs/encoder_{}_weights.pt".format(epoch)))
        decoder.load_state_dict(torch.load("saved_runs/decoder_{}_weights.pt".format(epoch)))
        accuracy = compute_model_accuracy(encoder, decoder, loader, device, epoch, writer)
        accuracies.append(accuracy)

    print("=" * 80)
    print("Final Accuracy = {:.1f}".format(100.*np.max(accuracies)))
    print("=" * 80)

    return accuracies


def compute_model_accuracy(encoder, decoder, loader, device, epoch, writer):
    num_correct = 0
    num_total = 0

    encoder.eval()
    decoder.eval()

    print("\rComputing validation accuracy model @ {} epoch..".format(epoch))

    for encoder_input, decoder_input, decoder_output in tqdm(loader):
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        decoder_output = decoder_output.to(device)

        with torch.no_grad():
            _, encoder_hidden = encoder(encoder_input)
            logits = decoder(decoder_input, encoder_hidden)
            predicted_sequence = logits.argmax(dim=-1)

        num_correct += (predicted_sequence == decoder_output).sum().item()
        num_total += decoder_output.shape[1]
    accuracy = (1. * num_correct) / float(num_total)
    writer.add_scalar("validation_accuracy", accuracy, epoch)
    return accuracy

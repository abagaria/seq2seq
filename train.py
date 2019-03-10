# Python imports.
import pickle
import argparse
import pdb
from tqdm import tqdm

# PyTorch imports.
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# Other imports.
from dataset import NMTDataset
from encoder import EncoderRNN
from decoder import DecoderRNN


def train(input_sentences, output_sentences, input_vocab, output_vocab, input_reverse, output_reverse, hy, writer):
    dataset = NMTDataset(input_sentences, output_sentences, input_vocab, output_vocab, input_reverse, output_reverse)
    loader = DataLoader(dataset, batch_size=hy.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_vocab_size = len(input_vocab.keys())
    output_vocab_size = len(output_vocab.keys())

    encoder = EncoderRNN(input_vocab_size, hy.embedding_size, hy.hidden_size, hy.rnn_layers, hy.bidirectional, device)
    decoder = DecoderRNN(output_vocab_size, hy.embedding_size, hy.hidden_size, hy.rnn_layers, hy.bidirectional, device)

    loss_function = nn.CrossEntropyLoss().to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=hy.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=hy.lr)

    n_iterations = 0
    loss_history = []
    training_accuracy = 0.

    encoder.train()
    decoder.train()

    for epoch in range(1, hy.num_epochs + 1):
        for encoder_input, decoder_input, decoder_output in tqdm(loader, desc="{}/{}".format(epoch, hy.num_epochs)):
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            decoder_output = decoder_output.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            _, encoder_hidden = encoder(encoder_input)
            logits = decoder(decoder_input, encoder_hidden)

            loss = loss_function(logits.view(hy.batch_size * decoder_output.shape[1], -1), decoder_output.view(-1))

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            writer.add_scalar("TrainingLoss", loss.item(), n_iterations)
            n_iterations = n_iterations + 1
            loss_history.append(loss.item())

        training_accuracy = compute_model_accuracy(encoder, decoder, loader, device, epoch, writer)
        torch.save(encoder.state_dict(), "saved_runs/encoder_{}_weights.pt".format(epoch))
        torch.save(decoder.state_dict(), "saved_runs/decoder_{}_weights.pt".format(epoch))

    return loss_history, training_accuracy


def compute_model_accuracy(encoder, decoder, loader, device, epoch, writer):
    num_correct = 0.
    num_total = 0.

    encoder.eval()
    decoder.eval()

    print("\rComputing training accuracy..")
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
    writer.add_scalar("training_accuracy", accuracy, epoch)

    encoder.train()
    decoder.train()

    return accuracy

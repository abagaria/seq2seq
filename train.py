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
from dataset import collate_fn
from model import Seq2Seq


def train(input_sentences, output_sentences, vocab, reverse_vocab, hy, writer):
    dataset = NMTDataset(input_sentences, output_sentences, vocab, reverse_vocab)
    loader = DataLoader(dataset, batch_size=hy.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab.keys())

    model = Seq2Seq(hy.embedding_size, hy.hidden_size, vocab_size, hy.rnn_layers, hy.bidirectional, device, dataset)

    loss_function = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hy.lr)

    n_iterations = 0
    loss_history = []
    training_accuracy = 0.

    model.train()

    for epoch in range(1, hy.num_epochs + 1):
        for encoder_input, encoder_len, decoder_input, decoder_input_len, decoder_output, decoder_output_len in \
                tqdm(loader, desc="{}/{}".format(epoch, hy.num_epochs)):

            # Move the data to the GPU
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            decoder_output = decoder_output.to(device)
            encoder_len = encoder_len.to(device)
            decoder_input_len = decoder_input_len.to(device)

            optimizer.zero_grad()

            logits = model(encoder_input, decoder_input, encoder_len, decoder_input_len)

            loss = loss_function(logits.view(hy.batch_size * decoder_output.shape[1], -1), decoder_output.view(-1))

            loss.backward()
            optimizer.step()

            writer.add_scalar("TrainingLoss", loss.item(), n_iterations)
            n_iterations = n_iterations + 1
            loss_history.append(loss.item())

        training_accuracy = compute_model_accuracy(model, loader, device, epoch, writer)
        torch.save(model.state_dict(), "saved_runs/seq2seq_{}_weights.pt".format(epoch))

    return loss_history, training_accuracy


def compute_model_accuracy(model, loader, device, epoch, writer):
    num_correct = 0.
    num_total = 0.

    model.eval()

    print("\rComputing training accuracy..")
    for encoder_input, encoder_len, decoder_input, decoder_input_len, decoder_output, decoder_output_len in loader:
        # Move the data to the GPU
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        decoder_output = decoder_output.to(device)
        encoder_len = encoder_len.to(device)
        decoder_input_len = decoder_input_len.to(device)

        with torch.no_grad():
            logits = model(encoder_input, decoder_input, encoder_len, decoder_input_len)
            predicted_sequence = logits.argmax(dim=-1)

        for i in range(encoder_input.shape[0]):
            output_length = decoder_output_len[i]
            predictions = predicted_sequence[i, :output_length]
            ground_truth = decoder_output[i, :output_length]

            num_correct += (predictions == ground_truth).sum().item()
            num_total += output_length

    accuracy = (1. * num_correct) / float(num_total)
    writer.add_scalar("training_accuracy", accuracy, epoch)

    model.train()

    return accuracy

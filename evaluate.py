# Python imports.
from tqdm import tqdm
import numpy as np
import pdb

# PyTorch imports.
import torch
from torch.utils.data import DataLoader

# Other imports.
from dataset import NMTDataset
from dataset import collate_fn
from model import Seq2Seq


def evaluate(input_sentences, output_sentences, vocab, reverse_vocab, hy, writer):
    dataset = NMTDataset(input_sentences, output_sentences, vocab, reverse_vocab)
    loader = DataLoader(dataset, batch_size=hy.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab.keys())

    model = Seq2Seq(hy.embedding_size, hy.hidden_size, vocab_size, hy.rnn_layers, hy.bidirectional, device, dataset)

    accuracies = []

    for epoch in range(1, hy.num_epochs + 1):
        model.load_state_dict(torch.load("saved_runs_batched/seq2seq_{}_weights.pt".format(epoch)))
        accuracy = compute_model_accuracy(model, loader, device, epoch, writer)
        accuracies.append(accuracy)

    print("=" * 80)
    print("Final Accuracy = {:.1f}".format(100.*np.max(accuracies)))
    print("=" * 80)

    return accuracies


def compute_model_accuracy(model, loader, device, epoch, writer):
    num_correct = 0
    num_total = 0

    model.eval()

    print("\rComputing validation accuracy model @ {} epoch..".format(epoch))

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

    writer.add_scalar("validation_accuracy", accuracy, epoch)

    return accuracy

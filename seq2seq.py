import sys

eng_filename = sys.argv[1]
frn_filename = sys.argv[2]


# TODO: Preprocess the dataset:
# - Shuffle the dataset
# - Split the dataset into 90-10
# - Tokenize the words into IDs
# - Pad sentences, and store sentence lengths
# - Create three outputs: Encoder input, decoder input, decoder output
# - Example encoder input: [hansard, revise, numero, 1, STOP]
# - Example decoder input: [START, edited, hansard, number, 1]
# - Example decoder output: [edited, hansard, number, 1, STOP]


# TODO: Create the sequence-to-sequence model. This should contain:
# - Embedding layers, one for french sentences and one for english sentences
# - An encoder network, taking in the french embeddings as input
# - A decoder network, taking in the english embeddings and encoded
#   french sentence as input
# - A loss function, comparing the decoder output with the expected decoder
#   output. (NOTE: make sure to mask the padded STOPs when computing the loss.)
# - A backpropagation function to minimize the loss
# - A function to calculate perplexity of the output
# - A function to calculate the accuracy of the output. The accuracy is
#   defined as the percentage of correct symbols.

# NOTE: In this file, you want to implement the seq2seq model for the
# BPE dataset. The model will be largely identical to the
# seq2seq model for the vanilla-dataset, except that there is only one
# word embedding for both the encoder (french) and decoder (english).
# This is due to the fact that the encoding is a joint-BPE.

# TODO: Train the model, then evaluate using the validation set.
# At the end, print out the final perplexity and accuracy score.

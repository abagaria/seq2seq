import sys
import re
from collections import defaultdict
from bpe import bpe
import pdb


def read_from_corpus(corpus_file):
    """
    Reads from the corpus, and returns english and french
    sentences.
    """
    eng_lines = []
    frn_lines = []

    with open(corpus_file, 'r') as f:
        for line in f:
            words = line.split("\t")
            eng_line = words[0]
            frn_line = words[1][:-1]

            eng_line = re.sub('([.,!?():;])', r' \1 ', eng_line)
            eng_line = re.sub('\s{2,}', ' ', eng_line)
            eng_line = eng_line.replace("\n", "")
            eng_line = eng_line.lower()

            frn_line = re.sub('([.,!?():;])', r' \1 ', frn_line)
            frn_line = re.sub('\s{2,}', ' ', frn_line)
            frn_line = frn_line.replace("\m", "")
            frn_line = frn_line.lower()

            eng_lines.append(eng_line.split())
            frn_lines.append(frn_line.split())
    return eng_lines, frn_lines


def count_vocabs(eng_lines, frn_lines):
    vocab = defaultdict(lambda: 0)
    inverse_vocab = defaultdict(lambda: "")

    for eng_line in eng_lines:
        for eng_word in eng_line:
            spaced_eng_word = " ".join(eng_word) + "</w>"
            vocab[spaced_eng_word] += 1
            inverse_vocab[eng_word] = spaced_eng_word
    for frn_line in frn_lines:
        for frn_word in frn_line:
            spaced_french_word = " ".join(frn_word) + "</w>"
            vocab[spaced_french_word] += 1
            inverse_vocab[frn_word] = spaced_french_word

    return vocab, inverse_vocab


def save_words(eng_lines, frn_lines, eng_filename, frn_filename):
    with open(eng_filename, 'w') as eng_file:
        for line in eng_lines:
            eng_file.write(' '.join(line) + '\n')

    with open(frn_filename, 'w') as frn_file:
        for line in frn_lines:
            frn_file.write(' '.join(line) + '\n')


def create_joint_file(new_english_lines, new_french_lines):
    with open("bpe-joint.txt", "w") as f:
        for line in new_english_lines:
            f.write(' '.join(line) + '\n')
        for line in new_french_lines:
            f.write(' '.join(line) + '\n')


# TODO: Implement byte pair encoding.
def replace(new_inverse_vocab, lines):
    new_lines = []
    for line in lines:  # type: list
        new_line = []
        for i, word in enumerate(line):
            if word in new_inverse_vocab:
                new_line.append(new_inverse_vocab[word])
            else:
                new_line.append(word)
            new_lines.append(new_line)
    return new_lines


if __name__ == '__main__':
    corpus_file = sys.argv[1]
    eng_filename = sys.argv[2]
    frn_filename = sys.argv[3]

    eng_lines, frn_lines = read_from_corpus(corpus_file)

    # TODO: Create a dictionary of words from the french and english corpus.
    count_vocab, inverse_vocab = count_vocabs(eng_lines, frn_lines)

    # TODO: Run your byte pair encoding function on the dictionary.
    new_count_vocab, new_inverse_vocab = bpe(count_vocab, inverse_vocab, num_merges=10000)

    # TODO: Generate new english and french lines to contain the byte-pair
    #       encoded words.
    new_english_lines = replace(new_inverse_vocab, eng_lines)
    new_french_lines = replace(new_inverse_vocab, frn_lines)

    # TODO: Save the english and french text, in separate files.
    save_words(new_english_lines, new_french_lines, eng_filename, frn_filename)
    create_joint_file(new_english_lines, new_french_lines)

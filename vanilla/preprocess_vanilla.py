import sys
import re
from collections import defaultdict


def count_vocabs(eng_lines, frn_lines):
    eng_vocab = defaultdict(lambda: 0)
    frn_vocab = defaultdict(lambda: 0)

    for eng_line in eng_lines:
        for eng_word in eng_line:
            eng_vocab[eng_word] += 1
    for frn_line in frn_lines:
        for frn_word in frn_line:
            frn_vocab[frn_word] += 1

    return eng_vocab, frn_vocab


def read_from_corpus(corpus_file):
    eng_lines = []
    frn_lines = []

    with open(corpus_file, 'r') as f:
        for line in f:
            words = line.split("\t")
            eng_line = words[0]
            frn_line = words[1][:-1]

            eng_line = re.sub('([.,!?():;])', r' \1 ', eng_line)
            eng_line = re.sub('\s{2,}', ' ', eng_line)
            frn_line = re.sub('([.,!?():;])', r' \1 ', frn_line)
            frn_line = re.sub('\s{2,}', ' ', frn_line)

            eng_line = [eng_word.lower() for eng_word in eng_line.split()]
            frn_line = [frn_word.lower() for frn_word in frn_line.split()]

            eng_lines.append(eng_line)
            frn_lines.append(frn_line)
    return eng_lines, frn_lines


def unk_words(eng_lines, frn_lines, eng_vocab, frn_vocab, threshold=5):
    for eng_line in eng_lines:
        for i in range(len(eng_line)):
            if eng_vocab[eng_line[i]] <= threshold:
                eng_line[i] = "UNK"

    for frn_line in frn_lines:
        for i in range(len(frn_line)):
            if frn_vocab[frn_line[i]] <= threshold:
                frn_line[i] = "UNK"


def save_words(eng_lines, frn_lines, eng_filename, frn_filename):
    with open(eng_filename, 'w') as eng_file:
        for line in eng_lines:
            eng_file.write(' '.join(line) + '\n')

    with open(frn_filename, 'w') as frn_file:
        for line in frn_lines:
            frn_file.write(' '.join(line) + '\n')


if __name__ == '__main__':
    corpus_file = sys.argv[1]
    eng_filename = sys.argv[2]
    frn_filename = sys.argv[3]

    eng_lines, frn_lines = read_from_corpus(corpus_file)
    eng_vocab, frn_vocab = count_vocabs(eng_lines, frn_lines)
    unk_words(eng_lines, frn_lines, eng_vocab, frn_vocab, 5)
    eng_vocab, frn_vocab = count_vocabs(eng_lines, frn_lines)
    save_words(eng_lines, frn_lines, eng_filename, frn_filename)

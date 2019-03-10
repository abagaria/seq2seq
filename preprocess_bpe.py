import sys
import re


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
            frn_line = re.sub('([.,!?():;])', r' \1 ', frn_line)
            frn_line = re.sub('\s{2,}', ' ', frn_line)

            eng_lines.append(eng_line.split())
            frn_lines.append(frn_line.split())
    return eng_lines, frn_lines


# TODO: Implement byte pair encoding.


if __name__ == '__main__':
    corpus_file = sys.argv[1]
    eng_filename = sys.argv[2]
    frn_filename = sys.argv[3]

    eng_lines, frn_lines = read_from_corpus(corpus_file)

    # TODO: Create a dictionary of words from the french and english corpus.

    # TODO: Run your byte pair encoding function on the dictionary.

    # TODO: Generate new english and french lines to contain the byte-pair
    #       encoded words.

    # TODO: Save the english and french text, in separate files.

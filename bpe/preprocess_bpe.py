import sys
import re
from collections import defaultdict
from bpe_procedure import bpe
import pdb
import pickle
import os


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


def count_vocabs(eng_lines, frn_lines, german_lines):
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
    for german_line in german_lines:
        for german_word in german_line:
            spaced_german_word = " ".join(german_word) + "</w>"
            vocab[spaced_german_word] += 1
            inverse_vocab[german_word] = spaced_german_word

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


def create_input_output_files(combined_file):
    input_file_name = combined_file.split(".")[0] + "-input-lines.txt"
    output_file_name = combined_file.split(".")[0] + "-output-lines.txt"
    print("Creating {} and {}".format(input_file_name, output_file_name))

    with open(combined_file, "r") as _file:
        input_lines = []
        output_lines = []
        for _line in _file:
            input_language = _line.split("\t")[0]
            output_language = _line.split("\t")[1][:-1]
            input_lines.append(input_language)
            output_lines.append(output_language)
    
    with open(input_file_name, "w+") as _file:
        for _line in input_lines:
            _file.write(_line + "\n")
    
    with open(output_file_name, "w+") as _file:
        for _line in output_lines:
            _file.write(_line + "\n")


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


def attach_target_token(input_lines, target_token="<2e>"):
    new_lines = []
    for line in input_lines:
        assert isinstance(line, list)
        new_line = [target_token] + line
        new_lines.append(new_line)
    return new_lines


def main():
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


if __name__ == '__main__':
    f2e_corpus = sys.argv[1]
    g2e_corpus = sys.argv[2]
    f2e_out_name = sys.argv[3]
    g2e_out_name = sys.argv[4]
    fg2e_out_name = sys.argv[5]

    f2e_eng_lines, f2e_frn_lines = read_from_corpus(f2e_corpus)
    g2e_eng_lines, g2e_gmn_lines = read_from_corpus(g2e_corpus)

    combined_eng_lines = f2e_eng_lines + g2e_eng_lines

    # TODO: Create a dictionary of words from both corpuses.
    count_vocab, inverse_vocab = count_vocabs(combined_eng_lines, f2e_frn_lines, g2e_gmn_lines)

    # TODO: Run your byte pair encoding function on the dictionary.
    new_count_vocab, new_inverse_vocab = bpe(count_vocab, inverse_vocab, num_merges=15000)

    # TODO: Generate new english, french and german lines to contain
    #       the byte-pair encoded words.
    new_f2e_eng_lines = replace(new_inverse_vocab, f2e_eng_lines)
    new_g2e_eng_lines = replace(new_inverse_vocab, g2e_eng_lines)
    new_f2e_frn_lines = replace(new_inverse_vocab, f2e_frn_lines)
    new_g2e_grm_lines = replace(new_inverse_vocab, g2e_gmn_lines)

    # TODO: For each of the new lines, append the necessary target
    #       language tokens. Also, shuffle the lines (make sure the
    #       shuffling scheme for the french lines is the same as
    #       the corresponding english lines - same for the german
    #       lines.)
    new_french_lines = attach_target_token(new_f2e_frn_lines)
    new_german_lines = attach_target_token(new_g2e_grm_lines)

    # TODO: Save three separate files. The first should contain
    #       french sentences along with the corresponding english.
    #       The second should contain german sentences along with the
    #       corresponding english. The third should contain both,
    #       alternating french and german sentences along with
    #       the corresponding english.
    with open(f2e_out_name, "w+") as f:
        for french_line, english_line in zip(new_french_lines, new_f2e_eng_lines):
            french_str = " ".join(french_line)
            english_str = " ".join(english_line)
            line = french_str + "\t" + english_str + "\n"
            f.write(line)

    with open(g2e_out_name, "w+") as f:
        for german_line, english_line in zip(new_german_lines, new_g2e_eng_lines):
            german_str = " ".join(german_line)
            english_str = " ".join(english_line)
            line = german_str + "\t" + english_str + "\n"
            f.write(line)

    with open(fg2e_out_name, "w+") as f:
        for f_line, g_line, ef_line, eg_line in zip(new_french_lines, new_german_lines, new_f2e_eng_lines, new_g2e_eng_lines):
            f_str = " ".join(f_line)
            g_str = " ".join(g_line)
            ef_str = " ".join(ef_line)
            eg_str = " ".join(eg_line)

            f2e_line = f_str + "\t" + ef_str + "\n"
            g2e_line = g_str + "\t" + eg_str + "\n"

            f.write(f2e_line)
            f.write(g2e_line)

    create_input_output_files(f2e_out_name)
    create_input_output_files(g2e_out_name)
    create_input_output_files(fg2e_out_name)

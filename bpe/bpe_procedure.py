import pdb
import re
from collections import defaultdict
from tqdm import tqdm


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in, v_in_inverse):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]

        original_word = "".join(word.split()).replace("</w>", "")
        v_in_inverse[original_word] = w_out
    return v_out, v_in_inverse


def bpe(vocab, inverse_vocab, num_merges=10):
    print("Original vocab size = ", len(vocab))
    for i in tqdm(range(num_merges)):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab, inverse_vocab = merge_vocab(best, vocab, inverse_vocab)
    print("BPE vocab size = ", len(vocab))
    return vocab, inverse_vocab


if __name__ == "__main__":
    v = {"l o w </w>": 5, "l o w e r </w>": 2,
         "n e w e s t </w>": 6, "w i d e s t </w>": 3}
    bpe(v)

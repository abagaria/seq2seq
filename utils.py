import random


def get_lines(input_file):
    with open(input_file, "r") as _file:
        lines = []
        for line in _file:
            line = line.replace("\n", "")
            lines.append(line)
    return lines


def create_data_splits(input_output_pairs):

    random.shuffle(input_output_pairs)

    num_lines = len(input_output_pairs)
    num_training_lines = int(0.9 * num_lines)

    training_pairs, validation_pairs = [], []

    for i, in_out_pair in enumerate(input_output_pairs):
        if i < num_training_lines:
            training_pairs.append(in_out_pair)
        else:
            validation_pairs.append(in_out_pair)
    print("Loaded {} training sentences and {} validation sentences".format(len(training_pairs),
                                                                            len(validation_pairs)))
    return training_pairs, validation_pairs

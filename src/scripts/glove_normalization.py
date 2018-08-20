import numpy as np
from argparse import ArgumentParser

DIMENSIONS = 50
WORD_NUMBER = 400000

HIGH = 1
LOW = -1


def min_max_scaling(vectors, low=LOW, high=HIGH):
    print("\nScaling vectors to min={} and max={}".format(low, high))
    minimum = np.min(vectors)
    maximum = np.max(vectors)
    rng = maximum - minimum

    return high - (((high - low) * (maximum - vectors)) / rng)


def student_z_standardizing(vectors):
    """
    See https://en.wikipedia.org/wiki/Standard_score
    """
    print("\nStandardizing vectors to a standard deviation of 1")
    v_mean = np.mean(vectors, axis=0, dtype=np.float64)
    v_std = np.std(vectors, axis=0, dtype=np.float64)

    norms = (vectors - v_mean) / v_std
    print("new vector mean: {}".format(np.mean(norms, axis=0)))
    print("new vector standard deviation: {}".format(np.std(norms, axis=0)))
    return norms.astype(float)


type_switcher = {
    'minmax': min_max_scaling,
    'studentz': student_z_standardizing
}


def main(input_file, output_file, dimensions, vocab_size, norm_type):
    embedding_vectors = np.zeros((vocab_size + 1, dimensions))
    word_numbers = {}

    # load embeddings
    print("Loading embedding vectors from {}".format(input_file))
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line_split = line.strip().split()
            word = line_split[0]
            vector = np.array(line_split[1:], dtype=float)

            word_numbers[word] = line_number
            embedding_vectors[line_number] = vector

    # normalize
    normalized_embedding_vectors = type_switcher[norm_type](embedding_vectors)

    # write to file
    print("Writing resulting vectors to embedding file: {}".format(output_file))
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in word_numbers:
            output = word

            line_number = word_numbers[word]
            embedding_vec = normalized_embedding_vectors[line_number]
            for value in embedding_vec:
                output += " " + str(value)

            output += "\n"

            f.write(output)


if __name__ == "__main__":
    arg_parse = ArgumentParser(description="Normalizes / Standardizes the word embedding vectors " +
                                           "for the glove embedding using min-max-scaling or " +
                                           "student-Z standardization.")
    arg_parse.add_argument('--dimensions',
                           type=int,
                           default=DIMENSIONS,
                           help="dimensionality of the embedding vector")
    arg_parse.add_argument('--vocab_size',
                           type=int,
                           default=WORD_NUMBER,
                           help="size of the vocabulary")
    arg_parse.add_argument('--type',
                           type=str,
                           default='minmax',
                           choices=type_switcher.keys(),
                           help="type of the normalization to perform on the vectors")
    arg_parse.add_argument('input',
                           type=str,
                           help="filepath to original glove word embedding file")
    arg_parse.add_argument('output',
                           type=str,
                           help="filepath, where the normalized glove embedding should be saved to")
    args = arg_parse.parse_args()

    main(args.input, args.output, args.dimensions, args.vocab_size, args.type)

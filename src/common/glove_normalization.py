import numpy as np

DIMENSIONS = 50
WORD_NUMBER = 400000
GLOVE_PATH_UNNORMALIZED = ''
GLOVE_PATH_NORMALIZED = ''

HIGH = 1
LOW = -1

embedding_vectors = np.zeros((WORD_NUMBER + 1, DIMENSIONS))
normalized_embedding_vectors = None
word_numbers = {}

# load embeddings
with open(GLOVE_PATH_UNNORMALIZED, 'r', encoding='utf-8') as f:
    for line_number, line in enumerate(f, 1):
        line_split = line.strip().split()
        word = line_split[0]
        vector = np.array(line_split[1:], dtype=float)

        word_numbers[word] = line_number
        embedding_vectors[line_number] = vector

# normalize
min = np.min(embedding_vectors)
max = np.max(embedding_vectors)
rng = max - min

normalized_embedding_vectors = HIGH - (((HIGH - LOW) * (max - embedding_vectors)) / rng)

#write to file
with open(GLOVE_PATH_NORMALIZED, 'w', encoding='utf-8') as f:
    output = ''
    for word in word_numbers:
        output = word

        line_number = word_numbers[word]
        embedding_vec = normalized_embedding_vectors[line_number]
        for value in embedding_vec:
            output += " " + str(value)

        output += "\n"

        f.write(output)


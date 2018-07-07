from src.common.dataloader.glove import Glove
from src.settings.settings import Settings

import numpy as np


def main():
    gl = Glove()
    gl.load_embedding()
    man_vec = gl.get_word_vector('man')
    # make random addition
    man_vec = np.add(man_vec, np.random.rand(*man_vec.shape))

    most_sim_word = gl.most_similar_word(man_vec)
    our_most_sim_word = gl.wordembedding_to_most_similar_word(man_vec)
    print("np.dot:")
    print(most_sim_word)
    print("our for:")
    print(our_most_sim_word)


if __name__ == "__main__":
    Settings.FILE = "../../settings/settings-sebastian.yml"
    main()

from src.common.dataloader.glove import Glove

gl = Glove()
gl.load_embedding()
gl.init_sims()
man_vec = gl.get_word_vector('man')
most_sim_word = gl.most_similar_word(man_vec)
print(most_sim_word)
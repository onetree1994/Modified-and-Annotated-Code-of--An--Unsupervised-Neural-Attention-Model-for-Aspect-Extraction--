import gensim
import codecs
import numpy as np

if __name__ == '__main__':
	np.set_printoptions(threshold = 10)
	model_file = '../preprocessed_data/restaurant/w2v_embedding'
	model = gensim.models.Word2Vec.load(model_file)
	print(model.wv['like', 'hello'])
	
	model_file = '../preprocessed_data/beer/w2v_embedding'
	model = gensim.models.Word2Vec.load(model_file)
	print(model.wv['like', 'hello'])
	
import gensim
import codecs

# 利用生成器读取数据可以避免等待数据读取的过程，速度更快.
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()

# 将数据转换为向量形式，基于word2vec算法.
def main(domain):
    source = '../preprocessed_data/%s/train.txt' % (domain)
    model_file = '../preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4)
    # try to print something to show the effectiveness of word2vec.
    print('model.mv:',model.wv['like'])
    model.save(model_file)

# 分别转换restaurant和beer数据集.
print ('Pre-training word embeddings ...')
main('restaurant')
main('beer')




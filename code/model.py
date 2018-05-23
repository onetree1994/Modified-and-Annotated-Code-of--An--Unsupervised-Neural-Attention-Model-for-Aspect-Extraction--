import logging
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def create_model(args, maxlen, vocab):

    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=-1, keepdims=True)), K.floatx())
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye((w_n.shape[0]).eval())))
        return args.ortho_reg*reg

    # 词汇表大小
    vocab_size = len(vocab)

    ##### Inputs #####
	# 正例的形状：batch_size * dim, 每个元素是在词汇表中的索引值, 每个句子有多少个词就有多少索引值
	# 负例的形状：batch_size * args.neg_size * dim, ditto
	# 得到w
    sentence_input = Input(batch_shape=(None, maxlen), dtype='int32', name='sentence_input')
    neg_input = Input(batch_shape=(None, args.neg_size, maxlen), dtype='int32', name='neg_input')

    ##### Construct word embedding layer #####
	# 嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]
	# keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    ##### Compute sentence representation #####
    # 计算句子嵌入，这里设计到keras的很多细节，日后还需要深入学习
    e_w = word_emb(sentence_input)
    y_s = Average()(e_w)
    att_weights = Attention(name='att_weights')([e_w, y_s])
    z_s = WeightedSum()([e_w, att_weights])

    ##### Compute representations of negative instances #####
    # 计算负例的z_n
    e_neg = word_emb(neg_input)
    z_n = Average()(e_neg)

    ##### Reconstruction #####
    # 重构过程
    p_t = Dense(args.aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(args.aspect_size, args.emb_dim, name='aspect_emb',
            W_regularizer=ortho_reg)(p_t)

    ##### Loss #####
    # 损失函数
    loss = MaxMargin(name='max_margin')([z_s, z_n, r_s])
    model = Model(input=[sentence_input, neg_input], output=loss)

    ### Word embedding and aspect embedding initialization ######
    # 如果定义了emb_path, 就用文件中的数值初始化E矩阵, T使用K-means初始化
    if args.emb_path:
        from w2vEmbReader import W2VEmbReader as EmbReader
        emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
        logger.info('Initializing word embedding matrix')
        model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))
        logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
        model.get_layer('aspect_emb').W.set_value(emb_reader.get_aspect_matrix(args.aspect_size))

    return model



    






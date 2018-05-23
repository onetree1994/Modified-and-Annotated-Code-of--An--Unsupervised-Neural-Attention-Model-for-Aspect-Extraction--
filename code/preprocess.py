from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs

# 移除停止词并提取名词主干.(提取名词主干，例如:'dogs'->'dog')
def parseSentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    # 将句子转化为list，并去掉标点符号.
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem

# 预处理训练数据.
def preprocess_train(domain):
    f = codecs.open('../datasets/'+domain+'/train.txt', 'r', 'utf-8')
    out = codecs.open('../preprocessed_data/'+domain+'/train.txt', 'w', 'utf-8')

    for line in f:
        tokens = parseSentence(line)
		# 确保处理后字符串长度大于0.
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n')

# 预处理测试数据.
def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('../datasets/'+domain+'/test.txt', 'r', 'utf-8')
    f2 = codecs.open('../datasets/'+domain+'/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('../preprocessed_data/'+domain+'/test.txt', 'w', 'utf-8')
    out2 = codecs.open('../preprocessed_data/'+domain+'/test_label.txt', 'w', 'utf-8')

    for text, label in zip(f1, f2):
        label = label.strip() # remove the spaces at the begin or at end of the string.
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        tokens = parseSentence(text)
		# 确保处理后字符串长度大于0.
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label+'\n')

def preprocess(domain):
    print ('\t'+domain+' train set ...')
    preprocess_train(domain)
    print ('\t'+domain+' test set ...')
    preprocess_test(domain)

print ('Preprocessing raw review sentences ...')
preprocess('restaurant')
preprocess('beer')



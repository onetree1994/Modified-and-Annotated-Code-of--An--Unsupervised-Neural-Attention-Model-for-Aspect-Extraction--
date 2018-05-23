from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs

# lmtzr.lemmatize(w)只修改名词，如需修改动词时态，需要lmtzr.lemmatize(w, 'v')

# remove stop words and lemmatize words of one sentence.(提取单词主干，例如:'loving'->'love')
if __name__ == '__main__':
    line = 'Bar was a little bit crowded , but these five girls know how to have fun ! ! it was a little hard to understand the waitress and she seemed to have little patience with our questions .'
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    print('[original ]', line)
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    print('[tokenize ]', text_token)
    text_rmstop = [i for i in text_token if i not in stop]
    print('[rmvstops ]', text_rmstop)
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    print('[lemmatize]', text_stem)
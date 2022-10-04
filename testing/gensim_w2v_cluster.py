# https://medium.com/@lukemenzies/using-machine-learning-to-perform-text-clustering-382ab33eb32a
# https://radimrehurek.com/gensim/models/word2vec.html
#
import re

#
import numpy
import pandas
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering

#
from preprocess import tokenize, exclude_stopwords, len_cut, lemma, stemma

#
data = pandas.read_csv('../headlines_en_clean.csv')

tokenized = data['text'].copy()
tokenized = tokenized.apply(func=tokenize)
tokenized = tokenized.str.lower()
tokenized = tokenized.apply(func=exclude_stopwords)
tokenized = tokenized.apply(func=len_cut)
tokenized = tokenized.apply(func=lemma)
# tokenized = tokenized.apply(func=stemma)


token = [to.split(' ') for to in tokenized.values.tolist()]
grams = Phrases(token, min_count=1, threshold=3, delimiter=' ')
phraser = Phraser(grams)

tokenized = []
for sent in token:
    tokenized.append(phraser[sent])

model = Word2Vec(tokenized, vector_size=10, min_count=5, epochs=40)
embedded = numpy.concatenate([numpy.array(model.wv[x]).mean(axis=0).reshape(1, -1) for x in tokenized], axis=0)

cluster = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average')


modelled = data[['text']].copy()
modelled['cluster'] = cluster.fit_predict(X=embedded)

# https://medium.com/@lukemenzies/using-machine-learning-to-perform-text-clustering-382ab33eb32a
# https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText
#
import re

#
import numpy
import pandas
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.fasttext import FastText
from sklearn.cluster import AgglomerativeClustering
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

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


tokenized = [to.split(' ') for to in tokenized.values.tolist()]

tokens = []
for i, sent in enumerate(tokenized):
    tokens.append(TaggedDocument(sent, [i]))

tokenized = list(tokens)
del tokens

model = Doc2Vec(vector_size=10, min_count=5, epochs=40)
model.build_vocab(tokenized)
model.train(tokenized, total_examples=model.corpus_count, epochs=model.epochs)

embedded = numpy.concatenate([numpy.array(model.infer_vector(x)).reshape(1, -1) for x in tokenized], axis=0)

cluster = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='average')


modelled = data[['text']].copy()
modelled['cluster'] = cluster.fit_predict(X=embedded)

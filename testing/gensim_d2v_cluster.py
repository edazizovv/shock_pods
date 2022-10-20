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
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

#
from preprocess import tokenize, exclude_stopwords, len_cut, lemma, stemma

#
#
# data = pandas.read_csv('../headlines_en_clean.csv')

# """
data = pandas.concat((pandas.read_csv('../headlines_en_clean.csv'),
                      pandas.read_csv('../breaking911_clean.csv')),
                     axis=0, ignore_index=True)
# data = data.iloc[:10000, :]
# """
# raise Exception("NATO")

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

model = Doc2Vec(vector_size=10, min_count=1, epochs=40)
# model = Doc2Vec(vector_size=10, min_count=1, epochs=100)   # weaker
# model = Doc2Vec(vector_size=5, min_count=1, epochs=40)     # weaker & kmeans stronger?
# model = Doc2Vec(vector_size=5, min_count=1, epochs=100)    # weaker
# model = Doc2Vec(vector_size=20, min_count=1, epochs=40)    # stronger?
# model = Doc2Vec(vector_size=20, min_count=1, epochs=100)   # weaker & kmeans stronger?

model.build_vocab(tokenized)
model.train(tokenized, total_examples=model.corpus_count, epochs=model.epochs)

embedded = numpy.concatenate([numpy.array(model.infer_vector(x.words)).reshape(1, -1) for x in tokenized], axis=0)


cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')   # 1->10093/0->3885

# cluster = DBSCAN(eps=0.50, metric='l2')  # 9535/3329/76/...
# cluster = DBSCAN(eps=0.75, metric='l2')  # 8954/3912/78/78/...
# cluster = DBSCAN(eps=0.90, metric='l2')  # 10694/2293/98/97/...
# cluster = DBSCAN(eps=0.50, metric='euclidean')  # 9267/3585/78/...
# cluster = DBSCAN(eps=0.75, metric='euclidean')  # 9103/3785/98/78/...

# cluster = KMeans(n_clusters=2, init='random', algorithm='lloyd')  # 8229/5749
# cluster = KMeans(n_clusters=2, init='random', algorithm='elkan')  # 8217/5761
# cluster = KMeans(n_clusters=2, init='k-means++', algorithm='lloyd')  # 8218/5760
# cluster = KMeans(n_clusters=2, init='k-means++', algorithm='elkan')  # 8163/5815

# cluster = SpectralClustering(n_clusters=2, affinity='poly', assign_labels='discretize')  # 12056/1922
# cluster = SpectralClustering(n_clusters=2, affinity='sigmoid', assign_labels='discretize')  # 10997/2981


modelled = data[['text']].copy()
modelled['cluster'] = cluster.fit_predict(X=embedded)

# modelled['cluster'].value_counts()

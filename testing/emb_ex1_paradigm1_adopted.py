# https://towardsdatascience.com/clustering-contextual-embeddings-for-topic-model-1fb15c45b1bd
#
import re

#
import numpy
import pandas
from umap import UMAP
from scipy import sparse
from simcse import SimCSE
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#
from preprocess import tokenize, exclude_stopwords, len_cut, lemma, stemma

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


tokenized = tokenized.values.tolist()
# '''
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
embeddings = model.encode(tokenized)
pandas.DataFrame(data=embeddings.numpy()).to_csv('./embeddings.csv', index=False)
# '''
# '''
# embeddings = pandas.read_csv('./embeddings.csv').values

embeddings_reduced = UMAP(n_neighbors=15,
                          n_components=10,
                          min_dist=0.0,
                          metric='cosine',
                          random_state=42
                          ).fit_transform(embeddings)

n_topics = 2

model_full = KMeans(n_topics)
model_full.fit(embeddings)
clusters_full = model_full.predict(embeddings)
embeddings_down = UMAP(n_neighbors=15,
                       n_components=2,
                       min_dist=0.0,
                       metric='cosine',
                       random_state=42
                       ).fit_transform(embeddings)
"""
model_reduced = KMeans(5)
model_reduced.fit(embeddings_reduced)
clusters_reduced = model_reduced.predict(embeddings_reduced)
embeddings_down_reduced = UMAP(n_neighbors=15,
                               n_components=2,
                               min_dist=0.0,
                               metric='cosine',
                               random_state=42
                               ).fit_transform(embeddings_reduced)
"""


class TopicWordsEstimator:

    def __init__(self, sub):
        self.sub = sub
        self.vectorizer = CountVectorizer()

    def estimate(self, cluster_data, all_data, all_data_clusters, cluster_no):
        self.vectorizer.fit(cluster_data)
        cluster_transformed = self.vectorizer.transform(cluster_data)
        all_transformed = self.vectorizer.transform(all_data)

        words = self.vectorizer.get_feature_names()
        scores, ix = self.sub.score(cluster_transformed=cluster_transformed,
                                    all_transformed=all_transformed, all_clusters=all_data_clusters,
                                    cluster_no=cluster_no)

        return scores, words, ix


class TT:

    # TODO: add other name options from the article
    def __init__(self, name):
        self.cluster_transformer = TfidfTransformer()
        self.all_transformer = TfidfTransformer()

    def score(self, cluster_transformed, all_transformed, all_clusters, cluster_no):
        all_transformed = self.all_transformer.fit_transform(all_transformed)
        all_tfidf = pandas.DataFrame(data=all_transformed.toarray())
        all_tfidf['topic'] = all_clusters
        all_tfidf_avg = all_tfidf.groupby(by='topic').mean()
        index = all_tfidf_avg.index.values.tolist().index(cluster_no)
        all_tfidf_avg = all_tfidf_avg.values

        self.cluster_transformer.fit(cluster_transformed)
        cluster_idfi = self.cluster_transformer.idf_

        scores = all_tfidf_avg * cluster_idfi
        scores = normalize(scores, axis=1, norm='l1', copy=False)
        scores = sparse.csr_matrix(scores)

        return scores, index


clusters = clusters_full.copy()

data['cluster'] = clusters

# clusters =
"""
cluster_sentences = [tokenized[j] for j in range(len(tokenized)) if clusters[j] == cluster]
top_estimator = TopicWordsEstimator(sub=TT(name=''))
scores, words = top_estimator.estimate(cluster_data=cluster_sentences, all_data=tokenized, all_data_clusters=clusters)
# '''
"""
if isinstance(tokenized[0], list):
    tokenized = [' '.join(token) for token in tokenized]
elif isinstance(tokenized[0], str):
    pass
else:
    raise Exception("Internal type error")

topics = {'cluster': [], 'words': []}
for cluster in numpy.unique(clusters):
    cluster_sentences = [tokenized[j] for j in range(len(tokenized)) if clusters[j] == cluster]
    topic_estimator = TopicWordsEstimator(sub=TT(name=''))
    scores, words, ix = topic_estimator.estimate(cluster_data=cluster_sentences, all_data=tokenized,
                                             all_data_clusters=clusters, cluster_no=cluster)
    scores = scores.toarray()[ix, :]
    scored = pandas.DataFrame(data={'scores': scores, 'words': words}).sort_values(
        by='scores', ascending=False)

    top = '; '.join(scored['words'].values[:10].tolist())
    topics['cluster'].append(cluster)
    topics['words'].append(top)
topics = pandas.DataFrame(data=topics)

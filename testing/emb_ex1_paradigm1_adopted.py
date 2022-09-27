# https://towardsdatascience.com/clustering-contextual-embeddings-for-topic-model-1fb15c45b1bd
#
import re

#
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


#
stop_words = stopwords.words('english')
data = pandas.read_csv('../headlines_en_clean.csv')


def is_number(symbol):
    numbers = '1234567890'
    return symbol in numbers


def is_symbol(symbol):
    symbols = ',.;:-?!/\\()"' + "'"
    return symbol in symbols


def is_spaces(symbol):
    spaces = ' \n'
    return symbol in spaces


def not_all_numbers(text):
    return not all([is_number(x) for x in text if (not is_symbol(x)) and (not is_spaces(x))])


def tokenize(text):
    text_wordlist = []
    for x in re.split(r'([.,!?\s]+)', text):
        if x and (x not in ['.', ' ', ', ', '. ']) and (x.lower() not in stop_words) and not_all_numbers(x):
            text_wordlist.append(x)
    return text_wordlist


tokenized = data['text'].apply(func=tokenize).values.tolist()
sentences = [' '.join(text_list) for text_list in tokenized]
'''
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
embeddings = model.encode(sentences)
pandas.DataFrame(data=embeddings.numpy()).to_csv('./embeddings.csv', index=False)
'''
# '''
embeddings = pandas.read_csv('./embeddings.csv').values

embeddings_reduced = UMAP(n_neighbors=15,
                          n_components=10,
                          min_dist=0.0,
                          metric='cosine',
                          random_state=42
                          ).fit_transform(embeddings)

model_full = KMeans(5)
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

    def estimate(self, cluster_data, all_data, all_data_clusters):
        self.vectorizer.fit(cluster_data)
        cluster_transformed = self.vectorizer.transform(cluster_data)
        all_transformed = self.vectorizer.transform(all_data)

        words = self.vectorizer.get_feature_names()
        scores = self.sub.score(cluster_transformed=cluster_transformed,
                                all_transformed=all_transformed, all_clusters=all_data_clusters)

        return scores, words


class TT:

    # TODO: add other name options from the article
    def __init__(self, name):
        self.cluster_transformer = TfidfTransformer()
        self.all_transformer = TfidfTransformer()

    def score(self, cluster_transformed, all_transformed, all_clusters):
        all_transformed = self.all_transformer.fit_transform(all_transformed)
        all_tfidf = pandas.DataFrame(data=all_transformed.toarray())
        all_tfidf['topic'] = all_clusters
        all_tfidf_avg = all_tfidf.groupby(by='topic').mean().values

        self.cluster_transformer.fit(cluster_transformed)
        cluster_idfi = self.cluster_transformer.idf_

        scores = all_tfidf_avg * cluster_idfi
        scores = normalize(scores, axis=1, norm='l1', copy=False)
        scores = sparse.csr_matrix(scores)

        return scores


cluster = 1

clusters = clusters_full.copy()
# clusters =

cluster_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == cluster]
top_estimator = TopicWordsEstimator(sub=TT(name=''))
scores, words = top_estimator.estimate(cluster_data=cluster_sentences, all_data=sentences, all_data_clusters=clusters)
# '''

#


#
import pandas
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#


#
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

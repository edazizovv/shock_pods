#


#
import pandas
from umap import UMAP
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import matthews_corrcoef
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering


#
from catapults import Trebuchet
from extensions import TopicWordsEstimator, TT
from mcc_adopt import mcc_adopted


#
# data = pandas.read_csv('../headlines_en_clean.csv')

# """
a = pandas.read_csv('../headlines_en_clean.csv')
a['category'] = '1'
b = pandas.read_csv('../breaking911_clean.csv')
b['category'] = '2'
data = pandas.concat((a,
                      b),
                     axis=0, ignore_index=True)
# data = data.iloc[:10000, :]
# """
# raise Exception("NATO")

run_code = None
save_embeddings = False

# embedder = 'w2v'
# embedder = 'ft'
# embedder = 'd2v'

# embedder_kwg = {'vector_size': 10, 'min_count': 1, 'epochs': 40}
# embedder_kwg = {'vector_size': 10, 'min_count': 1, 'epochs': 100}
# embedder_kwg = {'vector_size': 5, 'min_count': 1, 'epochs': 40}
# embedder_kwg = {'vector_size': 5, 'min_count': 1, 'epochs': 100}
# embedder_kwg = {'vector_size': 20, 'min_count': 1, 'epochs': 40}
# embedder_kwg = {'vector_size': 20, 'min_count': 1, 'epochs': 100}

reducer = None
# reducer = 'umap'

reducer_kwg = None
# reducer_kwg = {'n_neighbors': 15, 'n_components': 10, 'min_dist': 0.0, 'metric': 'cosine'}

down = 'umap'

down_kwg = {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.0, 'metric': 'cosine'}

# cluster = AgglomerativeClustering
# cluster_kwg = {'n_clusters': 2, 'affinity': 'euclidean', 'linkage': 'ward'}

# cluster = KMeans
# cluster_kwg = {'n_clusters': 2, 'init': 'random', 'algorithm': 'full'}
# cluster_kwg = {'n_clusters': 2, 'init': 'random', 'algorithm': 'elkan'}
# cluster_kwg = {'n_clusters': 2, 'init': 'k-means++', 'algorithm': 'full'}
# cluster_kwg = {'n_clusters': 2, 'init': 'k-means++', 'algorithm': 'elkan'}

topic_estimator = 'twe'

topic_estimator_kwg = {'sub': TT(name='')}

topic_top_n = 10

trebuchet = Trebuchet(run_code=run_code,
                      embedder=embedder, embedder_kwg=embedder_kwg,
                      reducer=reducer, reducer_kwg=reducer_kwg,
                      down=down, down_kwg=down_kwg,
                      cluster=cluster, cluster_kwg=cluster_kwg,
                      topic_estimator=topic_estimator, topic_estimator_kwg=topic_estimator_kwg, topic_top_n=topic_top_n,
                      save_embeddings=save_embeddings)

data, topics = trebuchet.project(data=data, text_field='text')
scored = trebuchet.score(data=data, scorer=mcc_adopted)
trebuchet.plot(data=data)

# data[data['category'] == '2']['cluster'].value_counts()

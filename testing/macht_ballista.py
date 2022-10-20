#


#
import pandas
from umap import UMAP
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import matthews_corrcoef
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering


#
from catapults import Ballista
from extensions import TopicWordsEstimator, TT
from mcc_adopt import mcc_adopted


#
if __name__ == '__main__':
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

    # topic_estimator = 'lda_gensim_single'
    topic_estimator = 'lda_gensim_ensemble'

    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 0.5, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 0.5, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 0.5, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 0.75, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 0.75, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 0.75, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 1, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 1, 'offset': 1}
    topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 1, 'offset': 1}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 0.5, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 0.5, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 0.5, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 0.75, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 0.75, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 0.75, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'auto', 'decay': 1, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'symmetric', 'decay': 1, 'offset': 10}
    # topic_estimator_kwg = {'num_topics': 2, 'passes': 100, 'alpha': 'asymmetric', 'decay': 1, 'offset': 10}

    topic_estimator_n = 10

    ballista = Ballista(run_code=run_code,
                        topic_estimator=topic_estimator, topic_estimator_kwg=topic_estimator_kwg,
                        topic_estimator_n=topic_estimator_n,
                        save_embeddings=save_embeddings)

    data, topics = ballista.project(data=data, text_field='text')
    scored = ballista.score(data=data, scorer=mcc_adopted)

    # data[data['category'] == '2']['topic'].value_counts()

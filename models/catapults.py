#
import os


#
# from packaging import version
# if version.parse(sklearn.__version__) < version.parse("1.0"):

import numpy
import pandas
import seaborn
from umap import UMAP
from simcse import SimCSE
from octis.models.CTM import CTM
from octis.dataset.dataset import Dataset
from topicx.baselines.cetopictm import CETopicTM
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec, FastText, Doc2Vec, EnsembleLda
#
from extensions import TopicWordsEstimator
from preprocess import tokenize, exclude_stopwords, len_cut, lemma, stemma


#
class Trebuchet:
    def __init__(self, run_code,
                 embedder, embedder_kwg, reducer, reducer_kwg, down, down_kwg, cluster, cluster_kwg,
                 topic_estimator, topic_estimator_kwg, topic_top_n,
                 save_embeddings=False,
                 tokenize=True, lower=True, stopwords=True, len_cut=True, lemma=True, stemma=False):

        self.run_code = run_code

        self.embedder_code = embedder
        self.embedder_kwg = embedder_kwg
        if embedder == 'w2v':
            self.embedder = Word2Vec
        elif embedder == 'ft':
            self.embedder = FastText
        elif embedder == 'd2v':
            self.embedder = Doc2Vec
        elif embedder == 'simcse':
            self.embedder = SimCSE
        else:
            raise KeyError("Invalid `embedder` keyword value provided: {0}".format(embedder))

        self.cluster = cluster
        self.cluster_kwg = cluster_kwg

        self.reducer_code = reducer
        self.reducer_kwg = reducer_kwg
        if reducer is None:
            self.reducer = None
        elif reducer == 'umap':
            self.reducer = UMAP
        else:
            raise KeyError("Invalid `reducer` keyword value provided: {0}".format(reducer))

        self.down_code = down
        self.down_kwg = down_kwg
        if down == 'umap':
            self.down = UMAP
        else:
            raise KeyError("Invalid `down` keyword value provided: {0}".format(down))

        self.topic_estimator_code = topic_estimator
        self.topic_estimator_kwg = topic_estimator_kwg
        self.topic_top_n = topic_top_n
        if topic_estimator == 'twe':
            self.topic_estimator = TopicWordsEstimator
        else:
            raise KeyError("Invalid `topic_estimator` keyword value provided: {0}".format(topic_estimator))

        self.save_embeddings = save_embeddings
        self.tokenize = tokenize
        self.lower = lower
        self.stopwords = stopwords
        self.len_cut = len_cut
        self.lemma = lemma
        self.stemma = stemma

    def project(self, data, text_field):

        x = data[text_field].copy()

        if self.save_embeddings:
            if os.path.isfile('./{0}.csv'.format(self.run_code)):
                embedded = pandas.read_csv('./embeddings.csv').values
            else:
                embedded = None
        else:
            embedded = None

        tokenized = self.preprocess(x=x)

        if embedded is None:

            if self.embedder_code in ['w2v', 'ft']:
                token = [to.split(' ') for to in tokenized.values.tolist()]
                grams = Phrases(token, min_count=1, threshold=3, delimiter=' ')
                phraser = Phraser(grams)

                tokenized = []
                for sent in token:
                    tokenized.append(phraser[sent])

                self.embedder = self.embedder(tokenized, **self.embedder_kwg)

                embedded = numpy.concatenate([numpy.array(self.embedder.wv[x]).mean(axis=0).reshape(1, -1)
                                              for x in tokenized], axis=0)
            elif self.embedder_code == 'd2v':
                tokenized = [to.split(' ') for to in tokenized.values.tolist()]

                tokens = []
                for i, sent in enumerate(tokenized):
                    tokens.append(TaggedDocument(sent, [i]))

                tokenized = list(tokens)
                del tokens

                self.embedder = self.embedder(**self.embedder_kwg)
                self.embedder.build_vocab(tokenized)
                self.embedder.train(tokenized, total_examples=self.embedder.corpus_count, epochs=self.embedder.epochs)

                embedded = numpy.concatenate(
                    [numpy.array(self.embedder.infer_vector(x.words)).reshape(1, -1) for x in tokenized], axis=0)
            elif self.embedder_code == 'simcse':
                tokenized = tokenized.values.tolist()

                self.embedder = self.embedder(**self.embedder_kwg)

                embedded = self.embedder.encode(tokenized)
            else:
                raise Exception("Internal key error")

            if self.save_embeddings:
                pandas.DataFrame(data=embedded).to_csv('./{0}.csv'.format(self.run_code), index=False)

        if self.reducer:
            self.reducer = self.reducer(**self.reducer_kwg)

            embedded = self.reducer.fit_transform(embedded)

        self.cluster = self.cluster(**self.cluster_kwg)
        clusters = self.cluster.fit_predict(embedded)

        self.down = self.down(**self.down_kwg)
        downed = self.down.fit_transform(embedded)

        data['cluster'] = clusters
        if downed.shape[1] == 2:
            data[['d1', 'd2']] = downed
        else:
            raise Exception("The second dimension size of downed embeddings should be equal to 2")

        if self.topic_estimator_code == 'twe':
            if isinstance(tokenized[0], list):
                tokenized = [' '.join(token) for token in tokenized]
            elif isinstance(tokenized[0], str):
                pass
            elif isinstance(tokenized[0], TaggedDocument):
                tokenized = [' '.join(token.words) for token in tokenized]
            else:
                raise Exception("Internal type error")

            topics = {'cluster': [], 'words': []}
            for cluster in numpy.unique(clusters):
                cluster_sentences = [tokenized[j] for j in range(len(tokenized)) if clusters[j] == cluster]
                topic_estimator = TopicWordsEstimator(**self.topic_estimator_kwg)
                scores, words, ix = topic_estimator.estimate(cluster_data=cluster_sentences, all_data=tokenized,
                                                             all_data_clusters=clusters, cluster_no=cluster)
                scores = scores.toarray()[ix, :]
                scored = pandas.DataFrame(data={'scores': scores, 'words': words}).sort_values(
                    by='scores', ascending=False)

                top = '; '.join(scored['words'].values[:self.topic_top_n].tolist())
                topics['cluster'].append(cluster)
                topics['words'].append(top)
            topics = pandas.DataFrame(data=topics)
        else:
            raise Exception("Internal key error")

        return data, topics

    @staticmethod
    def score(data, scorer):

        return scorer(y_true=data['category'].values, y_pred=data['cluster'].values)

    @staticmethod
    def plot(data):

        seaborn.scatterplot(data=data, x="d1", y="d2", hue="cluster")

    def preprocess(self, x):

        tokenized = x.copy()
        if self.tokenize:
            tokenized = tokenized.apply(func=tokenize)
        if self.lower:
            tokenized = tokenized.str.lower()
        if self.stopwords:
            tokenized = tokenized.apply(func=exclude_stopwords)
        if self.len_cut:
            tokenized = tokenized.apply(func=len_cut)
        if self.lemma:
            tokenized = tokenized.apply(func=lemma)
        if self.stemma:
            tokenized = tokenized.apply(func=stemma)
        return tokenized

    @staticmethod
    def topic_cloud(topic):
        raise NotImplementedError()


class Ballista:
    def __init__(self, run_code,
                 topic_estimator, topic_estimator_kwg, topic_estimator_n,
                 save_embeddings=False,
                 tokenize=True, lower=True, stopwords=True, len_cut=True, lemma=True, stemma=False):

        self.run_code = run_code

        self.topic_estimator_code = topic_estimator
        if 'num_topics' in topic_estimator_kwg.keys():
            self.topic_estimator_kwg = topic_estimator_kwg
        else:
            raise KeyError("Invalid `topic_estimator_kwg` keys: should contain `num_topics` keyword, now contains {0}".
                           format(list(topic_estimator_kwg.keys())))
        self.topic_estimator_n = topic_estimator_n
        if topic_estimator == 'lda_gensim_single':
            self.topic_estimator = LdaModel
        elif topic_estimator == 'lda_gensim_ensemble':
            self.topic_estimator = EnsembleLda
        else:
            raise KeyError("Invalid `topic_estimator` keyword value provided: {0}".format(topic_estimator))

        self.save_embeddings = save_embeddings
        self.tokenize = tokenize
        self.lower = lower
        self.stopwords = stopwords
        self.len_cut = len_cut
        self.lemma = lemma
        self.stemma = stemma

    def project(self, data, text_field):

        x = data[text_field].copy()

        tokenized = self.preprocess(x=x)

        token = [to.split(' ') for to in tokenized.values.tolist()]
        grams = Phrases(token, min_count=1, threshold=3, delimiter=' ')
        phraser = Phraser(grams)

        tokenized = []
        for sent in token:
            tokenized.append(phraser[sent])

        dictionary = corpora.Dictionary(tokenized)
        corpus = [dictionary.doc2bow(text) for text in tokenized]

        if self.topic_estimator_code == 'lda_gensim_single':
            self.topic_estimator = self.topic_estimator(corpus=corpus, id2word=dictionary, **self.topic_estimator_kwg)
            keys = data.index.values.tolist()
            topics = {}
            for i in range(len(corpus)):
                cori = corpus[i]
                topic = self.topic_estimator.get_document_topics(cori, 0)
                probs = []
                for topic_id, topic_prob in topic:
                    probs.append(topic_prob)
                topics[keys[i]] = probs

            modelled = pandas.DataFrame.from_dict(topics, orient='index')
            topic_column_names = ['topic_{0}'.format(i) for i in range(self.topic_estimator_kwg['num_topics'])]
            modelled.columns = topic_column_names
            modelled['text'] = data['text'].copy()

            def find_topic(row, thresh=0.9):
                if thresh == max:
                    return row.loc[row == row.max()].index[0]
                else:
                    if (row.loc[row > thresh]).any():
                        return row.loc[row > thresh].index[0]
                    else:
                        return None

            def find_propensity(row, thresh=0.9):
                if thresh == max:
                    return row.loc[row == row.max()].values[0]
                else:
                    if (row.loc[row > thresh]).any():
                        return row.loc[row > thresh].values[0]
                    else:
                        return None

            modelled['topic'] = modelled.iloc[:, :self.topic_estimator_kwg['num_topics']].apply(
                find_topic, args=(max,), axis=1)
            modelled['propensity'] = modelled.iloc[:, :self.topic_estimator_kwg['num_topics']].apply(
                find_propensity, args=(max,), axis=1)
            modelled = modelled.drop(columns=modelled.columns[:self.topic_estimator_kwg['num_topics']])

            cols = modelled.columns
            add = [x for x in data.columns if x not in cols]
            modelled[add] = data[add].copy()

            data = modelled.copy()

            got_topics = self.topic_estimator.show_topics(num_topics=self.topic_estimator_kwg['num_topics'],
                                                          num_words=self.topic_estimator_n, formatted=False)
            topics = {'topic': list(range(len(got_topics))), 'words': ['; '.join([item[0] for item in topic[1]])
                                                                       for topic in got_topics]}
        elif self.topic_estimator_code == 'lda_gensim_ensemble':
            self.topic_estimator = self.topic_estimator(corpus=corpus, id2word=dictionary, **self.topic_estimator_kwg)
            num_topics = len(self.topic_estimator.print_topics())
            self.topic_estimator = self.topic_estimator.generate_gensim_representation()
            keys = data.index.values.tolist()
            topics = {}
            for i in range(len(corpus)):
                cori = corpus[i]
                topic = self.topic_estimator.get_document_topics(cori, 0)
                probs = []
                for topic_id, topic_prob in topic:
                    probs.append(topic_prob)
                topics[keys[i]] = probs

            modelled = pandas.DataFrame.from_dict(topics, orient='index')
            topic_column_names = ['topic_{0}'.format(i) for i in range(num_topics)]
            modelled.columns = topic_column_names
            modelled['text'] = data['text'].copy()

            def find_topic(row, thresh=0.9):
                if thresh == max:
                    return row.loc[row == row.max()].index[0]
                else:
                    if (row.loc[row > thresh]).any():
                        return row.loc[row > thresh].index[0]
                    else:
                        return None

            def find_propensity(row, thresh=0.9):
                if thresh == max:
                    return row.loc[row == row.max()].values[0]
                else:
                    if (row.loc[row > thresh]).any():
                        return row.loc[row > thresh].values[0]
                    else:
                        return None

            modelled['topic'] = modelled.iloc[:, :num_topics].apply(find_topic, args=(max,), axis=1)
            modelled['propensity'] = modelled.iloc[:, :num_topics].apply(find_propensity, args=(max,), axis=1)
            modelled = modelled.drop(columns=modelled.columns[:num_topics])

            cols = modelled.columns
            add = [x for x in data.columns if x not in cols]
            modelled[add] = data[add].copy()

            data = modelled.copy()

            got_topics = self.topic_estimator.show_topics(num_topics=num_topics,
                                                          num_words=self.topic_estimator_n, formatted=False)
            topics = {'topic': list(range(len(got_topics))), 'words': ['; '.join([item[0] for item in topic[1]])
                                                                       for topic in got_topics]}
        else:
            raise Exception("Internal key error")

        topics = pandas.DataFrame(data=topics)
        return data, topics

    def preprocess(self, x):

        tokenized = x.copy()
        if self.tokenize:
            tokenized = tokenized.apply(func=tokenize)
        if self.lower:
            tokenized = tokenized.str.lower()
        if self.stopwords:
            tokenized = tokenized.apply(func=exclude_stopwords)
        if self.len_cut:
            tokenized = tokenized.apply(func=len_cut)
        if self.lemma:
            tokenized = tokenized.apply(func=lemma)
        if self.stemma:
            tokenized = tokenized.apply(func=stemma)
        return tokenized

    @staticmethod
    def score(data, scorer):

        return scorer(y_true=data['category'].values, y_pred=data['topic'].values)


class Mangonel:
    def __init__(self, run_code,
                 topic_estimator, topic_estimator_kwg, topic_estimator_n,
                 save_embeddings=False,
                 tokenize=True, lower=True, stopwords=True, len_cut=True, lemma=True, stemma=False):

        self.run_code = run_code

        self.topic_estimator_code = topic_estimator
        self.topic_estimator_kwg = topic_estimator_kwg
        self.topic_estimator_n = topic_estimator_n
        if topic_estimator == 'cet':
            self.topic_estimator = CETopicTM
        elif topic_estimator == 'ctm':
            self.topic_estimator = CTM
        else:
            raise KeyError("Invalid `topic_estimator` keyword value provided: {0}".format(topic_estimator))

        self.save_embeddings = save_embeddings
        self.tokenize = tokenize
        self.lower = lower
        self.stopwords = stopwords
        self.len_cut = len_cut
        self.lemma = lemma
        self.stemma = stemma

    def project(self, data, text_field):

        x = data[text_field].copy()

        tokenized = self.preprocess(x=x)

        dictionary = list(corpora.Dictionary(tokenized.str.split(' ')).values())
        with open('./dataset/vocabulary.txt', 'w') as f:
            for line in dictionary:
                f.write('{0}\n'.format(line))

        dataset = Dataset()
        dataset.load_custom_dataset_from_folder("./dataset")

        if self.topic_estimator_code == 'cet':
            self.topic_estimator = self.topic_estimator(dataset=dataset, top_words=self.topic_estimator_n,
                                                        **self.topic_estimator_kwg)
            self.topic_estimator.train()
            topics = self.topic_estimator.get_topics()
            topics = {'topic': list(topics.keys()),
                      'words': '; '.join([x[0] for t in topics.keys() for x in topics[t]])}
            data['topic'] = self.topic_estimator.topics[:int(len(self.topic_estimator.topics) / 3)]
        elif self.topic_estimator_code == 'ctm':
            self.topic_estimator = self.topic_estimator(**self.topic_estimator_kwg)
            output = self.topic_estimator.train_model(dataset=dataset, top_words=self.topic_estimator_n)
            topics = output['topics']
            topics = {'topic': list(range(len(topics))), 'words': ['; '.join(x) for x in topics]}
            data['propensity'] = numpy.max(output['topic-document-matrix'], axis=0)
            data['topic'] = numpy.argmax(output['topic-document-matrix'], axis=0)
        else:
            raise Exception("Internal key error")

        topics = pandas.DataFrame(data=topics)
        return data, topics

    def preprocess(self, x):

        tokenized = x.copy()
        if self.tokenize:
            tokenized = tokenized.apply(func=tokenize)
        if self.lower:
            tokenized = tokenized.str.lower()
        if self.stopwords:
            tokenized = tokenized.apply(func=exclude_stopwords)
        if self.len_cut:
            tokenized = tokenized.apply(func=len_cut)
        if self.lemma:
            tokenized = tokenized.apply(func=lemma)
        if self.stemma:
            tokenized = tokenized.apply(func=stemma)
        return tokenized

    @staticmethod
    def topic_cloud(topic):
        raise NotImplementedError()

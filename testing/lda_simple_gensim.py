# https://towardsdatascience.com/nlp-topic-modeling-to-identify-clusters-ca207244d04f
#
import re

#
import numpy
import pandas
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.phrases import Phrases, Phraser

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

dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(text) for text in tokenized]

n_topics = 4
model = LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary, passes=100, alpha='auto')

for i, topic in enumerate(model.print_topics(5)):
    print('{0}: {1}\n'.format(i + 1, topic))

keys = data.index.values.tolist()
topics = {}
for i in range(len(corpus)):
    cori = corpus[i]
    topic = model.get_document_topics(cori, 0)
    probs = []
    for topic_id, topic_prob in topic:
        probs.append(topic_prob)
    topics[keys[i]] = probs

modelled = pandas.DataFrame.from_dict(topics, orient='index')
topic_column_names = ['topic_{0}'.format(i) for i in range(0, n_topics)]
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


modelled['topic'] = modelled.iloc[:, 0:(n_topics-1)].apply(find_topic, args=(max,), axis=1)
modelled['propensity'] = modelled.iloc[:, 0:(n_topics-1)].apply(find_propensity, args=(max,), axis=1)
modelled.drop(columns=modelled.columns[:n_topics], inplace=True)

magi = modelled.groupby(by='topic')['propensity'].describe()

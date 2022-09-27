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

token = [to for to in tokenized]
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

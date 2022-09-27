# https://towardsdatascience.com/nlp-topic-modeling-to-identify-clusters-ca207244d04f
#
import re

#
import numpy
import pandas
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

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

dictionary = corpora.Dictionary(tokenized)

corpus = numpy.array([' '.join(token) for token in tokenized])
cvt = CountVectorizer(ngram_range=(1, 3), min_df=0.0, max_df=1.0)
corpus = pandas.DataFrame(data=cvt.fit_transform(corpus).toarray(), columns=cvt.get_feature_names())

n_topics = 10
model = LatentDirichletAllocation(
                                 learning_decay=0.7,
                                 max_doc_update_iter=100,
                                 max_iter=10,
                                 mean_change_tol=0.001,
                                 n_components=n_topics)
param_grid = {'n_components': [2, 3, 4, 5, 10],
              'learning_decay': [0.5, 0.7, 0.9]}

cv = GridSearchCV(estimator=model, param_grid=param_grid)
cv.fit(X=corpus)

feature_names = cvt.get_feature_names()

model = cv.best_estimator_

n_top_words = 10

topic_list = []
for topic_idx, topic in enumerate(model.components_):
    top_n = [feature_names[i]
             for i in topic.argsort()
             [-n_top_words:]][::-1]

    top_features = '; '.join(top_n)

    topic_list.append(f"topic_{'_'.join(top_n[:3])}")

    print(f"Topic {topic_idx}: {top_features}")
    print('\n\n\n')


modelled = data[['text']].copy()
topic_result = model.transform(corpus)
modelled['topic'] = topic_result.argmax(axis=1)
modelled['propensity'] = topic_result.max(axis=1)

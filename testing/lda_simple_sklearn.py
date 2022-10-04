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


corpus = tokenized.values
cvt = CountVectorizer(ngram_range=(1, 3), min_df=0.0, max_df=1.0)
corpus = pandas.DataFrame(data=cvt.fit_transform(corpus).toarray(), columns=cvt.get_feature_names())

n_topics = 4
model = LatentDirichletAllocation(n_components=n_topics)
model.fit(X=corpus)

feature_names = cvt.get_feature_names()

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

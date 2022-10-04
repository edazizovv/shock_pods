# https://towardsdatascience.com/clustering-contextual-embeddings-for-topic-model-1fb15c45b1bd
#
import re


#
import numpy
import pandas
from simcse import SimCSE
from gensim import corpora
from nltk.corpus import stopwords
from octis.models.CTM import CTM
from octis.dataset.dataset import Dataset

#
from preprocess import tokenize, exclude_stopwords, len_cut, lemma, stemma

#
data = pandas.read_csv('../headlines_en_clean.csv')


tokenized = data['text'].copy()
tokenized = tokenized.apply(func=tokenize)
tokenized = tokenized.str.lower()
tokenized = tokenized.apply(func=exclude_stopwords)
tokenized = tokenized.apply(func=len_cut)
# tokenized = tokenized.apply(func=lemma)
tokenized = tokenized.apply(func=stemma)


sentences = tokenized.values.tolist()

model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
embeddings = model.encode(sentences)
pandas.DataFrame(data=embeddings.numpy()).to_csv('./embeddings.csv', index=False)

# '''
embeddings = pandas.read_csv('./embeddings.csv').values


dataset = pandas.DataFrame(data={'document': sentences})
dataset['partition'] = 'train'
_dataset = dataset.copy()
_dataset['partition'] = 'val'
__dataset = dataset.copy()
__dataset['partition'] = 'test'
dataset = pandas.concat((dataset, _dataset, __dataset), axis=0, ignore_index=True)
dataset.to_csv('./dataset/corpus.tsv', sep='\t', index=False, header=False)
dictionary = list(corpora.Dictionary(tokenized.str.split(' ')).values())
with open('./dataset/vocabulary.txt', 'w') as f:
    for line in dictionary:
        f.write('{0}\n'.format(line))


dataset = Dataset()
dataset.load_custom_dataset_from_folder("./dataset")

model = CTM(num_topics=5, inference_type='combined', bert_model='princeton-nlp/sup-simcse-bert-base-uncased')
# model = CTM(num_topics=5, inference_type='combined', bert_model='bert-base-nli-mean-tokens')
output = model.train_model(dataset)
topics = output['topics']

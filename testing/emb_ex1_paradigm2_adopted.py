# https://towardsdatascience.com/clustering-contextual-embeddings-for-topic-model-1fb15c45b1bd
#
import re


#
import numpy
import pandas
from gensim import corpora
from nltk.corpus import stopwords
from octis.models.CTM import CTM
from octis.dataset.dataset import Dataset

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


def replace_symbols(text):
    result = ''
    for x in text:
        if not is_symbol(x):
            result += x
    return result


tokenized = data['text'].apply(func=replace_symbols).apply(func=tokenize).values.tolist()
sentences = [' '.join(text_list) for text_list in tokenized]
'''
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
embeddings = model.encode(sentences)
pandas.DataFrame(data=embeddings.numpy()).to_csv('./embeddings.csv', index=False)
'''
# '''
embeddings = pandas.read_csv('./embeddings.csv').values

'''
dataset = pandas.DataFrame(data={'document': sentences})
dataset['partition'] = 'train'
_dataset = dataset.copy()
_dataset['partition'] = 'val'
__dataset = dataset.copy()
__dataset['partition'] = 'test'
dataset = pandas.concat((dataset, _dataset, __dataset), axis=0, ignore_index=True)
dataset.to_csv('./dataset/corpus.tsv', sep='\t', index=False, header=False)
dictionary = list(corpora.Dictionary(tokenized).values())
with open('./dataset/vocabulary.txt', 'w') as f:
    for line in dictionary:
        f.write('{0}\n'.format(line))
'''

dataset = Dataset()
dataset.load_custom_dataset_from_folder("./dataset")

model = CTM(num_topics=5, inference_type='combined', bert_model='princeton-nlp/sup-simcse-bert-base-uncased')
# model = CTM(num_topics=5, inference_type='combined', bert_model='bert-base-nli-mean-tokens')
output = model.train_model(dataset)
topics = output['topics']

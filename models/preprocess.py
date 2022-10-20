# https://www.datacamp.com/tutorial/stemming-lemmatization-python
#
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer


# SOME CLEANING

# gensim.parsing.preprocessing.STOPWORDS
punctuations = "?:!.,;()-\'\""

def tokenize(text):
    text_wordlist = [x for x in word_tokenize(text) if x not in punctuations]
    return ' '.join(text_wordlist)

# STOPWORDS

stop_words = stopwords.words('english')

def exclude_stopwords(text):
    text_wordlist = [x for x in text.split(' ') if x not in stop_words]
    return ' '.join(text_wordlist)

# LEMMA

wordnet_lemmatizer = WordNetLemmatizer()

def lemma(text):
    text_wordlist = [wordnet_lemmatizer.lemmatize(x) for x in text.split(' ')]
    return ' '.join(text_wordlist)

# STEMMA

stemmer = SnowballStemmer("english", ignore_stopwords=True)
def stemma(text):
    text_wordlist = [stemmer.stem(x) for x in text.split(' ')]
    return ' '.join(text_wordlist)


# REMOVE BY LEN

def len_cut(text, thresh=3):
    text_wordlist = [x for x in text.split(' ') if len(x) > thresh]
    return ' '.join(text_wordlist)

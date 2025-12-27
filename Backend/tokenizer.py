import re
from nltk.corpus import stopwords
from nltk import PorterStemmer

# Exact same stopwords from GCP notebook
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    """
    Tokenize text and remove stopwords.
    Matches the tokenization from GCP notebook exactly.
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in all_stopwords]

def tokenize_stemmed(text):
    stemmer = PorterStemmer()
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    stemmed = [stemmer.stem(token) for token in tokens if token not in all_stopwords]
    return stemmed


def og_tokenize(query):
    english_sw = frozenset(stopwords.words('english'))
    corpus_sw = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_sw = english_sw.union(corpus_sw)
    RE_W = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_W.finditer(query.lower())]
    return [token for token in tokens if token not in all_sw]
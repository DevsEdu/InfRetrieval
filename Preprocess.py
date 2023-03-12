import spacy
nlp = spacy.load("pt_core_news_sm")
import pt_core_news_sm
nlp = pt_core_news_sm.load()

import unicodedata

def preprocess(token):
    if not token.is_stop:
        if not token.is_punct:
            if not token.is_space:
                lemma = token.lemma_.lower()
                lemma = unicodedata.normalize('NFKD', lemma).encode('ascii', 'ignore').decode('ascii')

                return lemma
    
    return -1
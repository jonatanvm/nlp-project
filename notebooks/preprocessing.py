import re
from time import time

import nltk
import pandas as pd
import spacy


def nltk_preprocess(dataframe):
    try:
        from nltk.corpus import stopwords
    except Exception as e:
        print(e)
        nltk.download()  # download stopwords
        from nltk.corpus import stopwords

    from nltk.stem.snowball import SnowballStemmer
    stop = set(stopwords.words("finnish"))
    stemmer = SnowballStemmer("finnish")

    def remove_parentheses(doc):
        return re.sub(r"\([^\)]*\)", "", doc)

    def remove_commas(doc):
        '''Remove punctuation that does not signal end of sentence:
        commas, colons, semicolons, hyphens'''
        return re.sub(r"[,:;-]", "", doc)

    def cleaning(doc):
        # remove parenthetical additions
        doc = remove_parentheses(doc)
        doc = remove_commas(doc)

        txt = [stemmer.stem(token) for token in doc.split() if token not in stop and isinstance(token, str)]
        if len(txt) > 2:
            return ' '.join(txt)

    t = time()

    txt = [cleaning(row) for row in dataframe['speech']]

    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

    df_clean = pd.DataFrame({'clean': txt})
    df_clean = df_clean.dropna().drop_duplicates()
    return df_clean


def spacy_preprocess(dataframe):
    # https://spacy.io/usage/models
    nlp = spacy.load('xx_ent_wiki_sm', disable=['ner', 'parser'])  # disabling Named Entity Recognition for speed

    def cleaning(doc):
        text = [token.lemma_ for token in doc if not token.is_stop]
        if len(text) > 2:
            return ' '.join(text)

    brief_cleaning = (re.sub("^([A-Z]|Å|Ä|Ö)[a-zåäö]+$", ' ', str(row)).lower() for row in dataframe['speech'])

    t = time()

    txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

    df_clean = pd.DataFrame({'clean': txt})
    df_clean = df_clean.dropna().drop_duplicates()
    return df_clean

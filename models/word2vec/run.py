import sys
import uuid

import pandas as pd
from gensim.models import Word2Vec

file_name = sys.argv[1] if len(sys.argv) >= 2 else None

if not file_name:
    df = pd.read_csv("output/speeches-1.csv", delimiter="|", lineterminator="\n")
    df = df[['speech']]
    sentences = [sentence[0].split(" ") for sentence in df[['speech']].values]
    print(sentences[:1])
    model = Word2Vec(sentences, min_count=1, workers=8)
    model.save(f"models/word2vec/word2vec-{uuid.uuid4().hex[:4]}.model")
else:
    model = Word2Vec.load(f"models/word2vec/{file_name}")

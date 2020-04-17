import csv

import pandas as pd

df = pd.read_csv("output/speeches-1.csv", delimiter="|", lineterminator="\n")
print(df.columns)
foo = lambda x: pd.Series([f"__label__{i.strip().replace(')', '').replace(' ', '-')}" for i in str(x).split('(')])
rev = df['speaker'].apply(foo)
print(rev)
df['speaker'] = rev[0].values
df['party'] = rev[1].values
df.drop('Unnamed: 0', axis=1, inplace=True)
df = df[['speaker', 'party', 'speech']]
print(df.head)
df.to_csv("models/fast_text/speeches.train", sep=" ", line_terminator="\n", index=False, quotechar=' ')

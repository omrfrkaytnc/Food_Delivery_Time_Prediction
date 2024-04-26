import pandas as pd

test = pd.read_csv('data/raw/test.csv')
train = pd.read_csv('data/raw/train.csv')

df = pd.concat([train, test])
dff = df.copy()



import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('Argumentation/admin.csv')
    df = df.drop(['id', 'Comments', 'label'], axis=1)
    df.to_excel('our_corpus.xlsx', index=False)
import pandas as pd

df_train = pd.read_excel('df_train.xlsx')
df_dev = pd.read_excel('df_dev.xlsx')
df_test = pd.read_excel('df_test.xlsx')

df = pd.concat([df_train, df_dev, df_test])
labels = df['Label']
labels_unique = list(set(labels))
for label in labels_unique:
    print('\"{}\",'.format(label), end=" ")
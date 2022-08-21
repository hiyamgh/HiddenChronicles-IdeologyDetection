import pandas as pd

df_train = pd.read_excel('df_train_single.xlsx')
df_dev = pd.read_excel('df_dev_single.xlsx')

df = pd.concat([df_train, df_dev])
labels = df['label']
labels_unique = list(set(labels))
for label in labels_unique:
    print('\"{}\",'.format(label), end=" ")
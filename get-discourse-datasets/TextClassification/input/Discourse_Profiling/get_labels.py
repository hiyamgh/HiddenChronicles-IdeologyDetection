import pandas as pd

df_train = pd.read_excel('df_train_cleaned.xlsx')
df_dev = pd.read_excel('df_dev_cleaned.xlsx')
df_test = pd.read_excel('df_test_cleaned.xlsx')

df = pd.concat([df_train, df_dev, df_test])
labels = df['Label']
labels_unique = list(set(labels))
for label in labels_unique:
    print('\"{}\",'.format(label), end=" ")
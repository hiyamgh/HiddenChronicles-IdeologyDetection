import pandas as pd
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('feature_extraction_train_updated.csv')
print(df_train.shape)
df_train, df_dev = train_test_split(df_train, test_size=0.1, random_state=42, stratify=list(df_train['label']))
df_train.to_csv('feature_extraction_train_updated_updated.csv', index=False)
df_dev.to_csv('feature_extraction_dev_updated_updated.csv', index=False)
print(df_train.shape)
print(df_dev.shape)

print('????????????????????????????????????????????????????????')
print((df_train['label'].value_counts()/len(df_train))*100)
print((df_dev['label'].value_counts()/len(df_dev))*100)
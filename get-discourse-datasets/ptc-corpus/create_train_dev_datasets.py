from dataset import *
import os
from sklearn.model_selection import train_test_split

print(os.getcwd())

train_data_folder = 'train-articles/'
dev_data_folder = 'dev-articles/'
labels_path_train = 'train-task-flc-tc.labels'
labels_path_dev = 'dev-task-flc-tc.labels'

train_articles, train_ref_articles_id, train_ref_span_starts, train_ref_span_ends, train_labels = load_data(train_data_folder, labels_path_train)
dev_articles, dev_ref_articles_id, dev_ref_span_starts, dev_ref_span_ends, dev_labels = load_data(dev_data_folder, labels_path_dev)

df_train = dataset_to_pandas(articles=train_articles, ref_articles_id=train_ref_articles_id,
                             ref_span_starts=train_ref_span_starts, ref_span_ends=train_ref_span_ends,
                             train_gold_labels=train_labels)

df_dev = dataset_to_pandas(articles=dev_articles, ref_articles_id=dev_ref_articles_id,
                             ref_span_starts=dev_ref_span_starts, ref_span_ends=dev_ref_span_ends,
                             train_gold_labels=dev_labels)

# print class percentages in each dataset
percentages_train = (df_train['label'].value_counts() / len(df_train)) * 100
percentages_dev = (df_dev['label'].value_counts() / len(df_dev)) * 100

print('\npercentages in training data:\n')
print(percentages_train)

print('\npercentages in dev data:\n')
print(percentages_dev)

# since percentages are not equal, will combine both datasets then do a stratified train/test split
df = pd.concat([df_train, df_dev]).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True) # shuffle dataframe

df_train, df_dev = train_test_split(df, test_size=len(df_dev), random_state=42, stratify=df['label'])
print('\nAFTER train/test split:\n')
percentages_train = (df_train['label'].value_counts() / len(df_train)) * 100
percentages_dev = (df_dev['label'].value_counts() / len(df_dev)) * 100

print('\npercentages in training data:\n')
print(percentages_train)

print('\npercentages in dev data:\n')
print(percentages_dev)

df_train.to_csv('df_train.csv', index=False)
df_dev.to_csv('df_dev.csv', index=False)
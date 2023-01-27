from dataset import load_data, get_train_dev_files, get_test_file
import os


train_data_folder = '../ptc-corpus (1)/train-articles/'
labels_path = '../ptc-corpus (1)/train-task-flc-tc.labels'
train_file = 'train.csv'
dev_file = 'dev.csv'
data_dir = 'annotations/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
articles, ref_articles_id, ref_span_starts, ref_span_ends, labels = load_data(train_data_folder, labels_path)
train_file_path = os.path.join(data_dir, train_file)
dev_file_path = os.path.join(data_dir, dev_file)
# if not os.path.exists(train_file_path) or not os.path.exists(dev_file_path):
print("Creating train/dev files: %s, %s", train_file_path, dev_file_path)
get_train_dev_files(articles, ref_articles_id, ref_span_starts, ref_span_ends, labels, train_file_path, dev_file_path)

import pandas as pd

df_train = pd.read_csv('annotations/train.csv')
df_val = pd.read_csv('annotations/dev.csv')

print('df_train.shape before dropping duplicates: {}'.format(df_train.shape))
df_train = df_train.drop_duplicates(subset=['context'])
df_train.to_csv(os.path.join(data_dir, 'train_multiclass.csv'), index=False)
print('df_train.shape after dropping duplicates: {}'.format(df_train.shape))


print('df_val.shape before dropping duplicates: {}'.format(df_val.shape))
df_val = df_val.drop_duplicates(subset=['context'])
df_val.to_csv(os.path.join(data_dir, 'dev_multiclass.csv'), index=False)
print('df_val.shape after dropping duplicates: {}'.format(df_val.shape))

count = 0
for label in list(set(df_val['label'])):
    print(label, end=';')
    count += 1
print(count)


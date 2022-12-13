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
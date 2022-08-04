import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

# import torch
# from torchtext.legacy import data
# from torchtext.vocab import Vectors, FastText
# import fasttext
# import spacy
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score
# import time
# from datetime import timedelta
#
#
# def get_time_dif(start_time):
#     end_time = time.time()
#     time_dif = end_time - start_time
#     return timedelta(seconds=int(round(time_dif)))
#
#
# class Dataset(object):
#     def __init__(self, config, text_column, label_column):
#         self.config = config
#         self.train_iterator = None
#         self.test_iterator = None
#         self.val_iterator = None
#         self.vocab = []
#         self.word_embeddings = {}
#         self.text_column = text_column # name of the column that contains the text data
#         self.label_column = label_column # name of the column that contains the labels
#
#     def get_pandas_df(self, filename):
#         '''
#         Load the data into Pandas.DataFrame object
#         This will be used to convert data to torchtext object
#         '''
#
#         df = pd.read_csv(filename) if '.csv' in filename else pd.read_excel(filename)
#         df = df[[self.text_column, self.label_column]]
#         return df
#
#     def get_vocab(self, df_train):
#         target_vocab = set()
#         for i, row in df_train.iterrows():
#             sentence = row[self.text_column]
#             sentence = sentence.split(' ')
#             for token in sentence:
#                 target_vocab.add(token)
#         return list(target_vocab)
#
#     def load_data(self, w2v_file, train_file, test_file=None, val_file=None, emb_dim=300):
#         '''
#         Loads the data from files
#         Sets up iterators for training, validation and test data
#         Also create vocabulary and word embeddings based on the data
#
#         Inputs:
#             w2v_file (String): path to file containing word embeddings (GloVe/Word2Vec)
#             train_file (String): path to training file
#             test_file (String): path to test file
#             val_file (String): path to validation file
#             emb_dim (int): embedding size
#         '''
#
#         # NLP = spacy.load('ar')
#         # tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
#         tokenizer = lambda x: x.split(' ')
#
#         # Creating Field for data
#         # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
#         TEXT = data.Field(sequential=True, tokenize=tokenizer, fix_length=self.config.max_sen_len) # will not user `lower=` since its in Arabic
#         LABEL = data.Field(sequential=False, use_vocab=False)
#         datafields = [(self.text_column, TEXT), (self.label_column, LABEL)]
#
#         # Load data from pd.DataFrame into torchtext.data.Dataset
#         train_df = self.get_pandas_df(train_file)
#         train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
#         train_data = data.Dataset(train_examples, datafields)
#
#         test_df = self.get_pandas_df(test_file)
#         test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
#         test_data = data.Dataset(test_examples, datafields)
#
#         # If validation file exists, load it. Otherwise get validation data from training data
#         if val_file:
#             val_df = self.get_pandas_df(val_file)
#             val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
#             val_data = data.Dataset(val_examples, datafields)
#         else:
#             print('validation dataset not given, therefore, will split the training data into 80% training and 20% for validation')
#             train_data, val_data = train_data.split(split_ratio=0.8)
#
#         if w2v_file:
#             target_vocab = self.get_vocab(df_train=train_df) # pass only the training data
#             print('length of target vocabulary: {}'.format(len(target_vocab)))
#             ft = fasttext.load_model('{}'.format(w2v_file))
#             matrix_len = len(target_vocab)
#             weights_matrix = np.zeros((matrix_len, 300))
#             words_found = 0
#
#             for i, word in enumerate(target_vocab):
#                 try:
#                     weights_matrix[i] = ft[word]
#                     words_found += 1
#                 except KeyError:
#                     weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
#
#         self.word_embeddings = weights_matrix
#
#         self.train_iterator = data.BucketIterator(
#             (train_data),
#             batch_size=self.config.batch_size,
#             sort_key=lambda x: len(x.text),
#             repeat=False,
#             shuffle=True)
#
#         self.val_iterator, self.test_iterator = data.BucketIterator.splits(
#             (val_data, test_data),
#             batch_size=self.config.batch_size,
#             sort_key=lambda x: len(x.text),
#             repeat=False,
#             shuffle=False)
#
#         print("Loaded {} training examples".format(len(train_data)))
#         print("Loaded {} test examples".format(len(test_data)))
#         print("Loaded {} validation examples".format(len(val_data)))
#
#
# if __name__ == '__main__':
#     cnf = Config()
#     dt = Dataset(config=cnf, text_column='context_ar', label_column='label')
#     dt.load_data(w2v_file='cc.ar.300.bin', train_file='../ptc-corpus/df_train_single.xlsx', test_file='../ptc-corpus/df_dev_single.xlsx')
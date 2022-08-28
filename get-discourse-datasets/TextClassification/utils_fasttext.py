import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(config):
    vocab_dic = {}
    tokenizer = lambda x: x.split(' ')
    df = pd.read_csv(config.train_path) if '.csv' in config.train_path else pd.read_excel(config.train_path)
    for _, row in df.iterrows():
        line = str(row[config.text_column])
        line = line.strip()

        if line.isdigit():
            continue

        for word in tokenizer(line):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items()], key=lambda x: x[1], reverse=True)
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    return vocab_dic


def get_word_embeddings(w2v, vocab):
    embeddings = np.random.rand(len(vocab), w2v.get_input_matrix().shape[1]) # `.get_input_matrix()` is specific to fasttext
    for word in vocab:
        idx = vocab[word]
        emb = w2v[word]
        embeddings[idx] = emb
    return embeddings


def build_dataset(config):

    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    vocab = build_vocab(config)

    def load_dataset(config, path, df=None):

        if df is None:
            df = pd.read_csv(path) if '.csv' in path else pd.read_excel(path)

        df = df[[config.text_column, config.label_column]]

        if not os.path.isfile('labels_ids.txt'):
            labels = list(set(df[config.label_column]))
            labels2id = {label: i for i, label in enumerate(labels)}
            with open('labels_ids.pkl', 'wb') as f:
                pickle.dump(labels2id, f)
        else:
            with open('labels_ids.pkl', 'rb') as f:
                labels2id = pickle.load(f)

        contents = []
        for i, row in df.iterrows():
            sentence = str(row[config.text_column]).strip()
            if sentence.isdigit():
                continue
            label = row[config.label_column]
            tokens = tokenizer(sentence)
            words_line = []

            if len(tokens) < config.max_sen_len:
                tokens.extend([PAD] * (config.max_sen_len - len(tokens)))
            else:
                tokens = tokens[:config.max_sen_len]

            for word in tokens:
                words_line.append(vocab.get(word, vocab.get(UNK)))

            contents.append((words_line, labels2id[label]))
        return contents, labels2id

    if config.dev_path is not None:
        # train, labels2id = load_dataset(config.train_path, config.pad_size)
        # dev, _ = load_dataset(config.dev_path, config.pad_size)
        # test, _ = load_dataset(config.test_path, config.pad_size)

        train, labels2id = load_dataset(config, config.train_path, df=None)
        dev, _ = load_dataset(config, config.dev_path, df=None)
        test, _ = load_dataset(config, config.test_path, df=None)
    else:
        # divide the training data into 80% training and 20% testing
        print('as `validation_path` is set to None, training data will be split into 80% training and 20% validation - stratified')
        df = pd.read_csv(config.train_path) if '.csv' in config.train_path else pd.read_excel(config.train_path)
        # df_train, df_val = train_test_split(df, test_size=0.20, random_state=42, stratify=list(df[config.label_column]))
        df_train, df_val = train_test_split(df, test_size=0.10, random_state=42, stratify=list(df[config.label_column]))

        # train, labels2id = load_dataset(config.train_path, config.pad_size, df_train)
        # dev, _ = load_dataset(config.dev_path, config.pad_size, df_val)
        # test, _ = load_dataset(config.test_path, config.pad_size)

        train, labels2id = load_dataset(config, config.train_path, df=df_train)
        dev, _ = load_dataset(config, config.dev_path, df=df_val)
        test, _ = load_dataset(config, config.test_path, df=None)

    return vocab, train, dev, test, labels2id


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
        # seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # return (x, seq_len), y
        return x, y

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
    """ get the time difference between given start time and current time """
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
        word_to_id = pickle.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pickle.dump(word_to_id, open(vocab_dir, 'wb'))

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
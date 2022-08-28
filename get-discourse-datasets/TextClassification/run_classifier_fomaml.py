# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import time
import torch
import numpy as np
from train_eval import train, init_network
from utils import build_dataset, build_iterator, get_time_dif
from importlib import import_module
import argparse
import fasttext
from utils import build_dataset
import random


class Config(object):

    """ Configuration parameters """
    # def __init__(self, dataset, embedding):
    def __init__(self, args):
        self.model_name = args.model
        self.train_path = args.training_path
        self.dev_path = args.validation_path
        self.test_path = args.testing_path
        self.text_column = args.text_column
        self.label_column = args.label_column
        self.save_path = 'saved_dict/'
        self.log_path = 'log/' + args.model + '/'
        self.embedding_path = args.embedding_path
        self.vocab_path = args.vocab_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = args.dropout # 0.5
        self.require_improvement = args.require_improvement # 1000
        self.class_list = None
        self.num_classes = None
        self.n_vocab = 0
        self.num_epochs = args.num_epochs # 1000
        self.batch_size = args.batch_size # 32
        # self.pad_size = args.pad_size # 32
        self.max_sen_len = args.max_sen_len
        self.learning_rate = args.lr # 1e-3

        if args.fasttext:
            self.embedding_pretrained = None
            self.embed = 300
        else:
            self.embedding_pretrained = torch.tensor(np.load(self.embedding_path)["embeddings"].astype('float32'))
            self.embed = self.embedding_pretrained.size(1)

        self.hidden_size = args.hidden_size # 128
        self.num_layers = args.num_layers # 2

        if self.model_name == 'TextCNN':
            self.filter_sizes = (2, 3, 4)
            self.num_filters = 256
        else:
            self.filter_sizes = None
            self.num_filters = None

        if self.model_name == 'TextRNN_Att':
            self.hidden_size2 = 64
        else:
            self.hidden_size2 = None
        self.pad_size = None


class RNNModel(nn.Module):
    def __init__(self, config):
        super(RNNModel, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x.long())  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out


def _sample_mini_dataset(dataset, num_classes, num_shots):
    random.shuffle(dataset)
    for cls in range(num_classes):
        cls_indices = [i for i, sentence_content in enumerate(dataset) if sentence_content[1] == cls]  # get indices of all examples pertaining to the class cls
        selected = np.random.choice(cls_indices, num_shots, replace=False)  # randomly select num_shots number of indices
        for i in selected:
            yield dataset[i]


def _mini_batches(samples, batch_size, num_batches, replacement):
    """
        Generate mini-batches from some data.
        :param samples: a mini-dataset containing num_shots*num_classes number of examples (per class)
        :param batch_size: size of a single batch
        :param num_batches: number of batches to generate
        :param replacement: boolean: whetehr to sample examples w/o replacement
        :return:
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return


def main():
    parser = argparse.ArgumentParser(description='Arabic Text Classification')
    parser.add_argument('--model', type=str, default='TextRNN',
                        help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
    parser.add_argument('--training_path', type=str, default='input/FAKES/feature_extraction_train_updated_updated.csv',
                        help='path to training dataset')
    parser.add_argument('--validation_path', type=str, default='input/FAKES/feature_extraction_dev_updated_updated.csv',
                        help='path to validation dataset')
    parser.add_argument('--testing_path', type=str, default='input/FAKES/feature_extraction_test_updated.csv',
                        help='path to testing dataset')
    parser.add_argument('--text_column', type=str, default='article_content',
                        help='name of the col containing text data inside train/val/test files')
    parser.add_argument('--label_column', type=str, default='label',
                        help='name of the col containing labels inside train/val/test files')
    parser.add_argument('--embedding_path', default='embeddings/fakes_embeddings.npz', type=str,
                        help='random or path to pre_trained embedding')
    parser.add_argument('--vocab_path', default='vocab/fakes_vocab.pkl', type=str,
                        help='path to the vocabulary mapping each word to an index')
    parser.add_argument('--fasttext', default=0, type=int,
                        help='whether to use fasttext embeddings or not. 0=False, 1=True')
    parser.add_argument('--max_sen_len', type=int, default=512, help='maximum length of sentence')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--require_improvement', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='number of batches')
    parser.add_argument('--pad_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=128, help='number of nodes in hidden layer')
    parser.add_argument('--num_layers', type=int, default=2, help='numer of hidden layers')
    parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')

    # FOMAML arguments
    parser.add_argument('--inner_iters', type=int, default=20, help='inner iterations')
    parser.add_argument('--inner_batch', type=int, default=5, help='inner batch size')
    parser.add_argument('--num_shots', type=int, default=5, help='number of examples per class')
    parser.add_argument('--meta_batch', type=int, default=1, help='meta training batch size')
    parser.add_argument('--meta_iters', type=int, default=700, help='meta-training iterations')
    parser.add_argument("--is_reptile", default=False, type=bool, help="whether use reptile or fomaml method")
    args = parser.parse_args()

    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    print('running FOMAML with {} as a base model ...'.format(model_name))
    training_file = args.training_path
    validation_file = args.validation_path
    testing_file = args.testing_path
    text_col = args.text_column
    label_col = args.label_column

    x = import_module('models.' + model_name)
    config = Config(args)

    np.random.seed(1)
    torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data, class_list = build_dataset(config, args.word)
    config.class_list = class_list
    config.num_classes = len(class_list)

    inner_iters = args.inner_iters  # number of inner iterations
    meta_iters = args.meta_iters  # number of meta train iterations
    num_shots = args.num_shots  # number of examples per class in the support set
    meta_batch = args.meta_batch  # meta_batch_size: number of tasks to sample **per meta iteration**
    N_class = config.num_classes  # number of classes in each task
    inner_batch = args.inner_batch  # ??
    Is_reptile = args.is_reptile

    num_train_optimization_steps = meta_iters * (num_shots) * N_class * meta_batch


if __name__ == '__main__':
    main()
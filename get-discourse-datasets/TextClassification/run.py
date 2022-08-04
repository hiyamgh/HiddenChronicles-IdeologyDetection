import time
import torch
import numpy as np
from train_eval import train, init_network
from utils import Dataset, get_time_dif
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Arabic Text Classification')
parser.add_argument('--model', type=str, default='TextRNN', help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--training_path', type=str, default='df_train_single.xlsx', help='path to training dataset')
parser.add_argument('--validation_path', type=str, default=None, help='path to validation dataset')
parser.add_argument('--testing_path', type=str, default='df_dev_single.xlsx', help='path to testing dataset')
parser.add_argument('--text_column', type=str, default='context_ar', help='name of the col containing text data inside train/val/test files')
parser.add_argument('--label_column', type=str, default='label', help='name of the col containing labels inside train/val/test files')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--max_sen_len', type=int, default=20, help='maximum length of sentence')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--require_improvement', type=int, default=1000)
parser.add_argument('--num_classes', type=int, default=14, help='number of classes (multi-class classification)')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='number of batches')
parser.add_argument('--pad_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden_size', type=int, default=128, help='number of nodes in hidden layer')
parser.add_argument('--num_layers', type=int, default=2, help='numer of hidden layers')
args = parser.parse_args()


class Config(object):

    """ Configuration parameters """
    # def __init__(self, dataset, embedding):
    def __init__(self, args, embeddings):
        # self.model_name = 'TextRNN'
        # self.train_path = dataset + '/data/train.txt'
        # self.dev_path = dataset + '/data/dev.txt'
        # self.test_path = dataset + '/data/test.txt'
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt', encoding='utf-8').readlines()]
        # self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = '/saved_dict/' + args.model + '.ckpt'
        self.log_path = '/log/' + args.model
        # self.embedding_pretrained = torch.tensor(
        #     np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
        #     if embedding != 'random' else None
        self.embedding_pretrained = embeddings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = args.dropout # 0.5
        self.require_improvement = args.require_improvement # 1000
        # self.num_classes = len(self.class_list)
        self.num_classes = args.num_classes
        # self.n_vocab = 0
        self.num_epochs = args.num_epochs # 1000
        self.batch_size = args.batch_size # 32
        self.pad_size = args.pad_size # 32
        self.learning_rate = args.lr # 1e-3
        # self.embed = self.embedding_pretrained.size(1)\
        #     if self.embedding_pretrained is not None else 300
        self.embed = self.embedding_pretrained.shape[1]
        self.hidden_size = args.hidden_size # 128
        self.num_layers = args.num_layers # 2


if __name__ == '__main__':
    embedding = '../cc.ar.300.bin'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    training_file = args.training_path
    validation_file = args.validation_path
    testing_file = args.testing_path
    text_col = args.text_column
    label_col = args.label_column

    dataset = Dataset(config=args, text_column=text_col, label_column=label_col)
    dataset.load_data(w2v_file=embedding, train_file=training_file, test_file=testing_file, val_file=validation_file, emb_dim=300)

    x = import_module('models.' + model_name)
    config = Config(args, dataset.word_embeddings)

    np.random.seed(1)
    torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    # train_iter = build_iterator(train_data, config)
    # dev_iter = build_iterator(dev_data, config)
    # test_iter = build_iterator(test_data, config)
    train_iter = dataset.train_iterator
    dev_iter = dataset.val_iterator
    test_iter = dataset.test_iterator
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
    # train(args, model, train_iter, dev_iter, test_iter)
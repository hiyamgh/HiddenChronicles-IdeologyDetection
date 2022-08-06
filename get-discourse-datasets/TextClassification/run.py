import time
import torch
import numpy as np
from train_eval import train, init_network
from utils import build_dataset, build_iterator, get_time_dif, get_word_embeddings
from importlib import import_module
import argparse
import fasttext

parser = argparse.ArgumentParser(description='Arabic Text Classification')
# parser.add_argument('--model', type=str, default='TextRNN', help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# parser.add_argument('--model', type=str, default='TextCNN', help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--model', type=str, default='TextRNN_Att', help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--training_path', type=str, default='df_train_single.xlsx', help='path to training dataset')
parser.add_argument('--validation_path', type=str, default=None, help='path to validation dataset')
parser.add_argument('--testing_path', type=str, default='df_dev_single.xlsx', help='path to testing dataset')
parser.add_argument('--text_column', type=str, default='context_ar', help='name of the col containing text data inside train/val/test files')
parser.add_argument('--label_column', type=str, default='label', help='name of the col containing labels inside train/val/test files')
parser.add_argument('--embedding', default='../cc.ar.300.bin', type=str, help='random or path to pre_trained embedding')
# parser.add_argument('--max_sen_len', type=int, default=20, help='maximum length of sentence')
parser.add_argument('--max_sen_len', type=int, default=128, help='maximum length of sentence')
# parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.8)
parser.add_argument('--require_improvement', type=int, default=1000)
# parser.add_argument('--num_classes', type=int, default=14, help='number of classes (multi-class classification)')
# parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='number of batches')
parser.add_argument('--pad_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden_size', type=int, default=128, help='number of nodes in hidden layer')
parser.add_argument('--num_layers', type=int, default=2, help='numer of hidden layers')
# parser.add_argument('--num_layers', type=int, default=3, help='numer of hidden layers')
args = parser.parse_args()


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

        self.embedding_pretrained = None
        self.embed = 300

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

if __name__ == '__main__':
    if args.embedding == 'random':
        embeddings = None
    else:
        embeddings = args.embedding
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    print('running {} ...'.format(model_name))
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
    vocab, train_data, dev_data, test_data, labels2id = build_dataset(config)
    config.class_list = list(labels2id.keys())
    config.num_classes = len(labels2id)

    # set these values as they were initially None
    if embeddings is not None:
        ft = fasttext.load_model(embeddings)
        embeddings_mod = get_word_embeddings(w2v=ft, vocab=vocab)
        config.embedding_pretrained = embeddings_mod
        config.embed = embeddings_mod.shape[1]

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    # if model_name != 'Transformer':
    #     init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
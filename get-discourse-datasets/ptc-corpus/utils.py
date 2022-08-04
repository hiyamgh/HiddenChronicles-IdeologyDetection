import torch
from torchtext.legacy import data
from torchtext.vocab import Vectors, FastText
import fasttext
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 32
    bidirectional = True
    output_size = 4
    max_epochs = 10
    lr = 0.25
    batch_size = 64
    max_sen_len = 20 # Sequence length for RNN
    dropout_keep = 0.8


class Dataset(object):
    def __init__(self, config, text_column, label_column):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
        self.text_column = text_column # name of the column that contains the text data
        self.label_column = label_column # name of the column that contains the labels

    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''

        # with open(filename, 'r') as datafile:
        #     data = [line.strip().split(',', maxsplit=1) for line in datafile]
        #     data_text = list(map(lambda x: x[1], data))
        #     data_label = list(map(lambda x: self.parse_label(x[0]), data))
        #
        # full_df = pd.DataFrame({"text": data_text, "label": data_label})
        # return full_df
        df = pd.read_csv(filename) if '.csv' in filename else pd.read_excel(filename)
        df = df[[self.text_column, self.label_column]]
        return df

    def get_vocab(self, df_train):
        target_vocab = set()
        for i, row in df_train.iterrows():
            sentence = row[self.text_column]
            sentence = sentence.split(' ')
            for token in sentence:
                target_vocab.add(token)
        return list(target_vocab)

    def load_data(self, w2v_file, train_file, test_file=None, val_file=None, emb_dim=300):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            w2v_file (String): path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        # NLP = spacy.load('ar')
        # tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        tokenizer = lambda x: x.split(' ')

        # Creating Field for data
        # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        TEXT = data.Field(sequential=True, tokenize=tokenizer, fix_length=self.config.max_sen_len) # will not user `lower=` since its in Arabic
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [(self.text_column, TEXT), (self.label_column, LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)

        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)

        if w2v_file:
            # vec = Vectors(w2v_file, cache='../', url='https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.bin.gz')
            # vec = Vectors(w2v_file, cache='../')
            # vec = FastText(w2v_file, cache='../')
            target_vocab = self.get_vocab(df_train=train_df) # pass only the training data
            print('length of target vocabulary: {}'.format(len(target_vocab)))
            ft = fasttext.load_model('{}'.format(w2v_file))
            # word_vectors = torch.from_numpy(ft.get_input_matrix())
            matrix_len = len(target_vocab)
            weights_matrix = np.zeros((matrix_len, 300))
            words_found = 0

            for i, word in enumerate(target_vocab):
                try:
                    weights_matrix[i] = ft[word]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
            # TEXT.build_vocab(train_data, vectors=Vectors(word_vectors))

            # TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
            # TEXT.build_vocab(train_data, vectors=w2v_file) # https://programs.wiki/wiki/torchtext-tutorial.html
        # self.word_embeddings = TEXT.vocab.vectors
        # self.vocab = TEXT.vocab

        self.word_embeddings = weights_matrix

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score


if __name__ == '__main__':
    cnf = Config()
    dt = Dataset(config=cnf, text_column='context_ar', label_column='label')
    dt.load_data(w2v_file='cc.ar.300.bin', train_file='df_train_single.xlsx', test_file='df_dev_single.xlsx')
    # https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ar.zip
    # dt.load_data(w2v_file='wiki.ar.vec', train_file='df_train_single.xlsx', test_file='df_dev_single.xlsx')
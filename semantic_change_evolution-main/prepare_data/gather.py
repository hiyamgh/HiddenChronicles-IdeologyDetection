import os
import fasttext
import pickle
import numpy as np
import random


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_common_vocab(start_year, end_year, models_path, save_path):
    '''
    :param start_year which year to start loading models from
    :param end_year: which year to stop loading models from (inclusive)
    :param models_path: path to the directory containing trained models
    :param save_path: path to the directory to save the common vocabulary
    '''

    print('Creating common vocabulary ...')
    years = list(range(start_year, end_year + 1))
    all_vocab = []
    for y in years:
        model = fasttext.load_model(os.path.join(models_path, '{}.bin'.format(y)))
        print('loaded model {}.bin'.format(y))
        all_vocab.append(model.words)

    # get the intersection of all vocabulary
    common_vocab = list(set.intersection(*map(set, all_vocab)))
    print('len of common vocab: {}'.format(len(common_vocab)))

    # save common vocab
    mkdir(save_path)
    with open(os.path.join(save_path, 'common_vocab.pkl'), 'wb') as f:
        pickle.dump(common_vocab, f)


def get_vectors_vocab(start_year, end_year, models_path, common_vocab_path, save_path):
    '''
    :param start_year which year to start loading models from
    :param end_year: which year to stop loading models from (inclusive)
    :param models_path: path to the directory containing trained models
    :param common_vocab_path: path to common vocabulary
    :param save_path: path to the directory to save the vectors
    '''
    print('\nCreating vocabulary vectors ...')
    # load the common vocab
    with open(os.path.join(common_vocab_path, 'common_vocab.pkl'), 'rb') as f:
        common_vocab = pickle.load(f)

    all_models = []
    all_years = list(range(start_year, end_year + 1))

    # so that we do not load models everytime, we will save the needed models into an array (i.e. load once)
    for y in all_years:
        all_models.append(fasttext.load_model(os.path.join(models_path, '{}.bin'.format(y))))
        print('added {}.bin'.format(y))

    # initialize the array (list of lists) for storing vectors
    # for each word, we wills tore its 10 representations (once per year)
    vectors = [[] for _ in range(len(common_vocab))]

    count = 0
    for w in common_vocab:
        for i in range(len(all_years)):
            vectors[count].append(all_models[i].get_word_vector(w))
        count += 1

    vectors = np.array(vectors)
    print('shape of all vectors: {}'.format(vectors.shape))

    # save the vectors data
    mkdir(save_path)
    with open(os.path.join(save_path, 'vectors.pkl'), 'wb') as f:
        pickle.dump(vectors, f)


def write_common_vocab(common_vocab_path, save_path):
    ''' writes common vocab to a txt file '''
    with open(os.path.join(common_vocab_path, 'common_vocab.pkl'), 'rb') as f:
        common_vocab = pickle.load(f)

    mkdir(save_path)
    with open(os.path.join(save_path, 'common_vocab.txt'), 'w', encoding='utf-8') as f:
        for w in common_vocab:
            f.write('{}\n'.format(w))
    f.close()


def create_train_test_indices(save_path):
    # get train and test indices
    with open(os.path.join(save_path, 'imp_words_test.txt'), 'r', encoding='utf-8') as f:
        words = f.readlines()
    imp_words = [w[:-1] if '\n' in w else w for w in words]
    for w in imp_words:
        print(w)
    with open(os.path.join(save_path, 'common_vocab.pkl'), 'rb') as f:
        common_vocab = pickle.load(f)

    # pick random 20% of the indices
    common_vocab_prime = [w for w in common_vocab if w not in imp_words]
    test_len = int(0.2 * len(common_vocab_prime)) + len(imp_words)
    train_len = len(common_vocab_prime) - test_len

    print('Length of training (80%): {}'.format(train_len))
    print('Length of testing  (20%): {}'.format(test_len))

    # generate train indices
    imp_idx = []
    for i, w in enumerate(common_vocab):
        if w in imp_words:
            # print('found {}'.format(w))
            imp_idx.append(i)
    # print(len(imp_idx))
    # print(len(imp_words))

    all_idx = list(range(len(common_vocab)))
    print(len(all_idx))
    all_idx = [idx for idx in all_idx if idx not in imp_idx]
    print(len(all_idx))
    train_idx = random.sample(all_idx, train_len)
    print(len(train_idx))
    test_idx = [idx for idx in all_idx if idx not in train_idx] + imp_idx
    print(len(test_idx))

    # save training and testing indices
    mkdir(save_path)
    with open(os.path.join(save_path, 'train_idx.pkl'), 'wb') as f:
        pickle.dump(train_idx, f)
    with open(os.path.join(save_path, 'test_idx.pkl'), 'wb') as f:
        pickle.dump(test_idx, f)
    with open(os.path.join(save_path, 'imp_idx.pkl'), 'wb') as f:
        pickle.dump(imp_idx, f)


if __name__ == '__main__':
    models_path = 'E:/fasttext_embeddings/ngrams4-size100-window3-mincount10-negative5-lr0.001/'
    save_path = '../data_proj/'
    start_year, end_year = 2000, 2009
    # # create and store common vocab
    # get_common_vocab(start_year=start_year, end_year=end_year, models_path=models_path, save_path=save_path)
    # # create and store vectors
    # get_vectors_vocab(start_year=start_year, end_year=end_year, models_path=models_path, common_vocab_path=save_path, save_path=save_path)
    # # write common vocab to a txt file
    # write_common_vocab(common_vocab_path=save_path, save_path=save_path)
    # create and save train/test indices
    create_train_test_indices(save_path=save_path)




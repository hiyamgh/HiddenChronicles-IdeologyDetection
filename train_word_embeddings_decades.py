import gensim
from gensim.models import Word2Vec
import tensorflow as tf
import os, re, string
import numpy as np
from utils import clean_text_arabic_style
from text_dirs import get_text_dirs
import time
from tensorboard.plugins import projector
import getpass
import logging
import h5py
import pickle
# Word2vec
import itertools
EMBEDDING_SIZE = 300
from train_word_embeddings import *

if __name__ == '__main__':
    # archive = 'hayat'
    # archive = 'assafir'
    archive = 'nahar'
    hf_location = None
    if getpass.getuser() == '96171':
        hf_location = 'E:/newspapers/'

    if getpass.getuser() == '96171':
        MODEL_DIR = 'E:/newspapers/word2vec/{}/embeddings/'.format(archive)
        MODEL_META = 'E:/newspapers/word2vec/{}/meta_data/'.format(archive)
        MODEL_DIR_OUT = 'E:/newspapers/word2vec_decades/{}/meta_data/'.format(archive)
    else:
        MODEL_DIR = 'D:/word2vec/{}/embeddings/'.format(archive)
        MODEL_META = 'D:/word2vec/{}/meta_data/'.format(archive)
        MODEL_DIR_OUT = 'D:/word2vec_decades/{}/meta_data/'.format(archive)

    mkdir(directory=MODEL_DIR)
    mkdir(directory=MODEL_META)
    mkdir(directory=MODEL_DIR_OUT)

    # documents = list()
    # tokenize = lambda x: gensim.utils.simple_preprocess(x)
    # t1 = time.time()
    # # docs = read_files(TEXT_DIRS, nb_docs=10000)
    # # docs, docs_by_year = read_files_hdf5(archive=archive, content_saving=MODEL_META, h5_location=hf_location)
    # read_files_hdf5(archive=archive, content_saving=MODEL_META, h5_location=hf_location)
    # t2 = time.time()
    # print('Reading docs took: {:.3f} mins'.format((t2 - t1) / 60))

    # items_per_bin = len(risk_df) // self.nb_bins
    #         bin_category = [0] * len(risk_df)
    #         for i in range(self.nb_bins):
    #             lower = i * items_per_bin
    #             if i != self.nb_bins - 1:
    #                 upper = (i + 1) * items_per_bin
    #                 bin_category[lower:upper] = [i] * (upper - lower)
    #             else:
    #                 bin_category[lower:] = [i] * (len(range(lower, len(risk_df))))

    print('\ngenerating word2vec on yearly docs - decade level ... ')
    all_content = []
    nb_years = 10
    if archive not in ['assafir', 'nahar']:
        years_found = [int(y[-6:-2]) for y in os.listdir(MODEL_META)]
    else:
        years_found = [int(y[-6:-2]) for y in os.listdir(MODEL_META)[:-1]]
    nb_groups = int(np.ceil(len(years_found) / nb_years))
    years_grouped = [[] for i in range(nb_groups)]
    for i in range(nb_groups):
        lower = i * nb_years
        if i != nb_groups - 1:
            upper = (i + 1) * nb_years
            years_grouped[i] = [y for y in years_found[lower:upper]]
        else:
            years_grouped[i] = [y for y in years_found[lower:]]
    for i, group in enumerate(years_grouped):

        if i != len(years_grouped) - 1:
            print('Training on decade: {}-{}'.format(group[0], group[-1]))
            grouped_content = []
            for year in group:
                with open(os.path.join(MODEL_META, 'content_{}.p'.format(year)), "rb") as fp:
                    print(year)
                    yearly_content = pickle.load(fp)
                    grouped_content.append(yearly_content)

            combined_content = list(itertools.chain(*grouped_content))
            model = gensim.models.Word2Vec(combined_content, size=EMBEDDING_SIZE, min_count=5)
            model.save(os.path.join(MODEL_DIR_OUT, 'word2vec_{}_{}'.format(str(group[0]), str(group[-1]))))
            print('finished training on docs of years {}-{}'.format(group[0], group[-1]))

    # with open(os.path.join(MODEL_META, 'content_2008.p'), "rb") as fp:
    #     # print(year)
    #     yearly_content = pickle.load(fp)
    #     print(yearly_content)

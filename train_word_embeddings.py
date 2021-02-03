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
EMBEDDING_SIZE = 300

tokenize = lambda x: gensim.utils.simple_preprocess(x)


def clean_doc(doc):
    """
    Cleaning a document by several methods
    """
    # Lowercase
    # doc = doc.lower()
    # Remove numbers
    doc = re.sub(r"[0-9]+", "", doc)
    # Split in tokens
    tokens = doc.split()
    # Remove punctuation
    tokens = [w.translate(str.maketrans('', '', string.punctuation)) for w in tokens]

    # Tokens with less then two characters will be ignored
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)


def read_files(text_directories, nb_docs):
    """
    Read in text files
    """
    documents = list()
    tokenize = lambda x: gensim.utils.simple_preprocess(x)
    print('started reading ...')
    for path in text_directories:
        count = 0
        # Read in all files in directory
        if os.path.isdir(path):
            all_files = os.listdir(path)
            for filename in all_files:
                if filename.endswith('.txt') and filename[0].isdigit():
                    count += 1
                    with open('%s/%s' % (path, filename), encoding='utf-8') as f:
                        doc = f.read()
                        doc = clean_text_arabic_style(doc)
                        doc = clean_doc(doc)
                        documents.append(tokenize(doc))
                        if count % 100 == 0:
                            print('processed {} files so far from {}'.format(count, path))
                if count >= nb_docs and count <= nb_docs + 200:
                    print('REACHED END')
                    break
        if count >= nb_docs and count <= nb_docs:
            print('REACHED END')
            break

    return documents


def read_files_hdf5(archive, content_saving, h5_location=None):
    ''' read files stored in hdf5 format which is way more optimized
        :param archive: the archive we want to train word embeddings on
                      : string
        :param content_saving: location where to save the contents (content
        of txt files saved pickled so we do not encounter memory errors)
                      :string
        :param h5_location: the location of the archives saved in hdf5 format
                      :string
    '''
    documents = list()

    if archive not in ['nahar', 'hayat', 'assafir']:
        raise ValueError('The requested archive is not found. You should choose one of the following: {}'.format(['nahar', 'hayat', 'assafir']))

    if h5_location is None:
        hf = h5py.File('{}.h5'.format(archive), 'r')
    else:
        hf = h5py.File('{}.h5'.format(os.path.join(h5_location, archive)), 'r')

    # reading all documents into one list
    # will cause memory error in python
    # better save a dictionary of each year
    # then concatenate lists from each year in the dictionary
    # documents = []
    # documents_by_year = {}
    count = 0
    # each group we defined is a year
    for group in hf.keys():
        issues_docs = []
        # if group not in documents_by_year:
        #     documents_by_year[group] = []
        print('-------------------------- group: {} ------------------------------------'.format(group))
        # loop over all datasets in a certain year -- datasets here are the actual issues
        for issue in hf[group].keys():
            doc = hf[group][issue].value
            # print('dest: {}'.format(dset))
            doc = clean_text_arabic_style(doc)
            doc = clean_doc(doc)

            content = tokenize(doc)

            # add the documents's data to list
            documents.append(content)

            # add the document's data to the yearly documents dictionary
            # documents_by_year[group].append(content)
            issues_docs.append(content)

            count += 1
            if count % 10000 == 0:
                print('processed {} files so far'.format(count))
        
        mkdir(directory=content_saving)
        file_name = "content_{}.p".format(group)
        with open(os.path.join(content_saving, file_name), "wb") as fp:  # Pickling
            pickle.dump(issues_docs, fp)

        print('wrote {} to {}'.format(file_name, content_saving))


def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), 300))

    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


def mkdir(directory):
    ''' create directory if it does not already exist '''
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':

    # archive = 'hayat'
    archive = 'assafir'
    hf_location = None
    if getpass.getuser() == '96171':
        hf_location = 'E:/newspapers/'

    if getpass.getuser() == '96171':
        MODEL_DIR = 'E:/newspapers/word2vec/{}/embeddings/'.format(archive)
        MODEL_META = 'E:/newspapers/word2vec/{}/meta_data/'.format(archive)
    else:
        MODEL_DIR = 'D:/word2vec/{}/embeddings/'.format(archive)
        MODEL_META = 'D:/word2vec/{}/meta_data/'.format(archive)

    mkdir(directory=MODEL_DIR)
    mkdir(directory=MODEL_META)

    # documents = list()
    # tokenize = lambda x: gensim.utils.simple_preprocess(x)
    t1 = time.time()
    # docs = read_files(TEXT_DIRS, nb_docs=10000)
    # docs, docs_by_year = read_files_hdf5(archive=archive, content_saving=MODEL_META, h5_location=hf_location)
    read_files_hdf5(archive=archive, content_saving=MODEL_META, h5_location=hf_location)
    t2 = time.time()
    print('Reading docs took: {:.3f} mins'.format((t2 - t1) / 60))

    print('\ngenerating word2vec on yearly docs ... ')
    all_content = []
    for file_name in os.listdir(MODEL_META):
        with open(os.path.join(MODEL_META, file_name), "rb") as fp:  # Pickling
            # all_content.append(pickle.load(fp))
            year = file_name.split('_')[1][:-2]
            yearly_content = pickle.load(fp)
            model = gensim.models.Word2Vec(yearly_content, size=EMBEDDING_SIZE, min_count=5)
            mkdir(MODEL_DIR)
            model.save(os.path.join(MODEL_DIR, 'word2vec_{}'.format(year)))
            print('finished training on docs of year {}'.format(year))
            all_content.append(yearly_content)

    import itertools
    word2vec_all_content = list(itertools.chain(*all_content))

    # print('Number of documents: %i' % len(docs))

    # # Training the model on all the documents
    # print('\nTraining word2vec on all docs')
    # model = gensim.models.Word2Vec(word2vec_all_content, size=EMBEDDING_SIZE, min_count=5)
    # mkdir(MODEL_DIR)
    # model.save(os.path.join(MODEL_DIR, 'word2vec_all'))
    # print('finished training on all docs')

    # print('\ngenerating word2vec on yearly docs ... ')
    # for year in docs_by_year:
    #     yearly_docs = docs_by_year[year]
    #     model = gensim.models.Word2Vec(docs, size=EMBEDDING_SIZE, min_count=5)
    #     mkdir(MODEL_DIR)
    #     model.save(os.path.join(MODEL_DIR, 'word2vec_{}'.format(year)))
    #     print('finished training on docs of year {}'.format(year))

    # visualize(model, MODEL_DIR)

    # model = Word2Vec.load('trained_models2/word2vec')
    # word_vectors = model.wv
    #
    # # get neighbors
    # neighbors = model.wv.most_similar(positive=['الفلسطينيه'], topn=50)
    # for neighbor in neighbors:
    #     print('{}, {:.3f}'.format(neighbor[0], neighbor[1]))
    #
    # # الفلسطيتيه, 0.825
    # # الفلسطينيهء, 0.823
    # # الفنسطينيه, 0.805
    # # الفتسطينيه, 0.786
    # # الفلسطبنيه, 0.696
    # # انفلسطينيه, 0.689
    # # لفلسطينيه, 0.686
    #
    # # generate all possible words from orig word using edt distance of 1: insertion + deletion
    #
    # print(model.similarity('الفلسطينيه', 'لبنان'))


# with open(os.path.join(MODEL_DIR,'metadata.tsv'), 'w', encoding='utf-8') as f:
#     f.writelines("\n".join(index_words))
#
# # define the model without training
# sess = tf.InteractiveSession()
#
# config = projector.ProjectorConfig()
# embedding = config.embeddings.add()
# embedding.tensor_name = 'embeddings'
# embedding.metadata_path = './metadata.tsv'
# # projector.visualize_embeddings(MODEL_DIR, config)
#
# tensor_embeddings = tf.Variable(model.wv.vectors, name='embeddings')
#
# checkpoint = tf.compat.v1.train.Saver([tensor_embeddings])
# checkpoint_path = checkpoint.save(sess=sess, global_step=None, save_path=os.path.join(MODEL_DIR, "model.ckpt"))
# projector.visualize_embeddings(MODEL_DIR, config)
#
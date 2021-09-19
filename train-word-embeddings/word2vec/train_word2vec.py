'''

We have stored all files in hdf5 format. This helped us store in each dataset in the file:
* The raw text
* The cleaned text after applying normalization
* meta data about the txt file stored (which is a page of a newspaper issue):
    - year
    - month
    - day
    - page number
This was done for each archive. For groupings we grouped by the year number. This will help
for navigation, and will take less time than looping over files in a directory in order to find
a certain issue in a certain year/day/ etc. Example of structure

____1995:
|_________95081102-r:   ... ذهب الولد الى المدرسسسه
          |__year
          |__month
          |__day
          |__pagenb

|_________95081109
          |__year
          |__month
          |__day
          |__pagenb
'''

from gensim.models import Word2Vec
import h5py, os, re, logging
import multiprocessing as mp
import argparse
import sys
sys.path.append('..')
from normalization import *


def mkdir(folder):
    ''' creates a directory if it doesn't already exist '''
    if not os.path.exists(folder):
        os.makedirs(folder)


class SentenceIterable(object):

    '''
    An iterable object. The difference between a generator and an iterable is that once you
    have passed through a generator once, you cannot pass anymore. However, iterable will create
    a new iterator every time its looped over.

    We needed a generator to generate sentences to overcome overloading RAM with all sentences
    at once. Therefore, we will pass the 'generated sentences' to this iterable class.

    The reason is that Word2vec need to have more than one pass over sentences, one for building
    vocabulary then another/other(s) for training.
    '''

    def __init__(self, hf, year):
        self.hf = hf
        self.year = year

    def __iter__(self):
        delimiters = en_FULL_STOP, ar_FULL_STOP
        # re.escape allows to build the pattern automatically and have the delimiters escaped nicely
        regexPattern = '|'.join(map(re.escape, delimiters))
        # define the Arabic Normalizer instance
        arabnormalizer = ArabicNormalizer()

        # sentences = []
        for issue in self.hf[self.year].keys():
            doc = self.hf[self.year][issue].value
            # lines = doc.readlines()
            lines = doc.split('\n')
            lines_cleaned = arabnormalizer.normalize_paragraph(lines)
            # store cleaned lines as a string (as if we re-stored a cleaned document back)
            doc_cleaned = ''
            for line in lines_cleaned:
                if line == '\n':
                    doc_cleaned += line
                else:
                    doc_cleaned += line + '\n'
            # get the sentences in the document (parts of the document separated by punctuation (mainly stop) marks)
            sentences = re.split(regexPattern, doc_cleaned)
            for sentence in sentences:
                sentence = sentence.replace('\n', '')
                sentence = sentence.strip()
                if sentence == '':
                    continue
                sentence = sentence.split(' ')
                # remove one letter words
                sentence = [s for s in sentence if len(s) > 1]
                yield sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--archive', type=str, default='assafir', help="name of the archive to transform")
    parser.add_argument('-s', '--size', type=int, default=100, help='dimension of word vectors')
    parser.add_argument('-w', '--window', type=int, default=5, help='size of the sliding window')
    parser.add_argument('-m', '--mincount', type=int, default=5, help='minimum number of occurences of a word to be considered')
    parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='number of worker threads to train the model')
    parser.add_argument('-g', '--sg', type=int, default=1, help='training algorithm: Skip-Gram (1), otherwise CBOW (0)')
    # parser.add_argument('-i', '--hs', type=int, default=1, help='use of hierachical sampling for training')
    parser.add_argument('-n', '--negative', type=int, default=0, help='use of negative sampling for training (usually between 5-20)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='initial learning rate')
    # parser.add_argument('-o', '--cbowmean', type=int, default=0, help='for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors')
    # args = parser.parse_args()
    args = parser.parse_args()

    print('processing {} archive'.format(args.archive))

    # get the location of the hdf5 file to open it
    hf = h5py.File('../../input/{}.h5'.format(args.archive), 'r')

    # get all years in the hdf5 file (each year is a group)
    years = list(hf.keys())

    logdir = '{}/{}/size{}-window{}-mincount{}-sg{}-negative{}-lr{}/'.format(args.archive, 'cbow' if args.sg == 0 else 'SGNS',
    args.size, args.window, args.mincount, args.sg, args.negative if args.sg == 1 else 0, args.learning_rate)

    # create the folder to save models, if it does not already exist
    if os.path.exists(logdir):
        pass
    else:
        mkdir(logdir)
        # train a model on each year in the data
        for year in years:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

            model = Word2Vec(sentences=SentenceIterable(hf, year), size=args.size, window=args.window,
                             min_count=args.mincount, workers=args.threads, sg=args.sg,
                             negative=args.negative if args.sg == 1 else 0,
                             alpha=args.learning_rate)
                             # cbow_mean=args.cbowmean)

            model.save(os.path.join(logdir, "model-{}.model".format(year)))
            # model.wv.save_word2vec_format(args.logdir, binary=False)



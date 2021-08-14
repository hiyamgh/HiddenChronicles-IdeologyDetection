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
import argparse
import sys
sys.path.append('..')
from normalization import *


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def SentenceGenerator(hf, year):
    # writing delimiters like this results in a tuple of unicode delimiters
    delimiters = EXCLAMATION, en_FULL_STOP, en_SEMICOLON, en_QUESTION, ar_FULL_STOP, ar_SEMICOLON, ar_QUESTION
    # re.escape allows to build the pattern automatically and have the delimiters escaped nicely
    regexPattern = '|'.join(map(re.escape, delimiters))
    # define the Arabic Normalizer instance
    arabnormalizer = ArabicNormalizer()

    for issue in hf[year].keys():
        doc = hf[year][issue].value
        lines = doc.readlines()
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
            yield sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=str, help="name of the archive to transform")
    parser.add_argument("logdir", type=str, help="name of teh directory to save trained models in")
    args = parser.parse_args()
    # get the location of the hdf5 file to open it
    hf = h5py.File('../../input/{}.h5'.format(args.archive), 'r')
    # get all years in the hdf5 file (each year is a group)
    years = list(hf.keys())
    # create the folder to save models, if it does not already exist
    mkdir(args.logdir)
    for year in years:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = SentenceGenerator(hf, year)
        model = Word2Vec(sentences=sentences, size=100, window=5, min_count=1, workers=4)
        model.save(os.path.join(args.logdir, "model-{}.model".format(year)))



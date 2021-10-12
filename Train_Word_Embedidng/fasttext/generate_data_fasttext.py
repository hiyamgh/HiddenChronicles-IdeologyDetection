'''
There are two implementations of FastText:
1- Facebook's FastText
2- Gensim's FastText

We will be using Facebook's FastText for the following reasons:

See my question and answer to it:
* https://stackoverflow.com/questions/68834211/difference-between-gensims-fasttext-and-facebooks-fasttext/

For a list of all hyper parameters related to gensim's FastText,
please see: https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText

However, the word-ngrams says:

word_ngrams (int, optional) – In Facebook’s FastText, “max length of word ngram”
- but gensim only supports the default of 1 (regular unigram word handling).

This means that Gensim's fasttext only supports unigrams, but not bigrams or trigrams,
So its a better option to use Facebook's FastText. For a list of Facebook's hyper parameters
please see:

* https://fasttext.cc/docs/en/options.html
* https://github.com/facebookresearch/fastText/blob/master/docs/unsupervised-tutorials.md

--------------------------------------------------------------------------------------------------
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

import h5py, os, re
import argparse
import sys
sys.path.append('..')
from normalization import *


def mkdir(folder):
    ''' creates a directory if it doesn't already exist '''
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_files(archive, hf, year, start_year=None, end_year=None):
    '''
    This method will gather all data related to a certain year in one file.
    This is because Facebook's implementation of Fasttext understands this form of input.
    Input need to be line by line. Therefore, we will be amending each sentence
    into a new line.

    :param archive: name of the archive
    :param hf: the hdf5 file of data per arcyhive per year per issue
    :param year: the year of interest
    ---- if we want to have a data file for several years collated together ----
    :param start_year: start year to get data from
    :param end_year: end year to get data from
    :return:
    '''
    delimiters = en_FULL_STOP, ar_FULL_STOP
    # re.escape allows to build the pattern automatically and have the delimiters escaped nicely
    regexPattern = '|'.join(map(re.escape, delimiters))
    # define the Arabic Normalizer instance
    arabnormalizer = ArabicNormalizer()

    if year is not None and start_year is None and end_year is None:

        print('started generating data files for {} archive at the yearly level...'.format(archive))
        data_folder = "data/{}/".format(archive)
        mkdir(data_folder)

        for issue in hf[year].keys():
            doc = hf[year][issue].value
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
            with open(os.path.join(data_folder, "{}.txt".format(year)), "a", encoding="utf-8") as f:
                for sentence in sentences:
                    sentence = sentence.replace('\n', '')
                    sentence = sentence.strip()
                    if sentence == '':
                        continue
                    sentence = sentence.split(' ')
                    # remove one letter words
                    sentence = [s for s in sentence if len(s) > 1]

                    # turn the sentence back to a string
                    sentence_str = " ".join(sentence)

                    # FastText accepts files that are line by line. Therefore we need to
                    # collect sentence by sentence and throw it into a txt file.
                    f.write(sentence_str + '\n')
            f.close()
    else:
        print('started generating data file for {} archive from year {} till year {}...'.format(archive, start_year, end_year))
        years_range = list(range(start_year, end_year + 1))
        data_folder = "data/{}/start_end/".format(archive)
        mkdir(data_folder)
        for year in years_range:
            for issue in hf[str(year)].keys():
                doc = hf[str(year)][issue].value
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
                with open(os.path.join(data_folder, "{}-{}.txt".format(start_year, end_year)), "a", encoding="utf-8") as f:
                    for sentence in sentences:
                        sentence = sentence.replace('\n', '')
                        sentence = sentence.strip()
                        if sentence == '':
                            continue
                        sentence = sentence.split(' ')
                        # remove one letter words
                        sentence = [s for s in sentence if len(s) > 1]

                        # turn the sentence back to a string
                        sentence_str = " ".join(sentence)

                        # FastText accepts files that are line by line. Therefore we need to
                        # collect sentence by sentence and throw it into a txt file.
                        f.write(sentence_str + '\n')
                f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--archive', type=str, default='assafir', help="name of the archive to transform")
    parser.add_argument('-fy', '--start_year', type=int, default=None, help='year to start from')
    parser.add_argument('-ty', '--end_year', type=int, default=None, help='year to end file on')
    args = parser.parse_args()

    print('processing {} archive'.format(args.archive))

    # get the location of the hdf5 file to open it
    hf = h5py.File('../../input/{}.h5'.format(args.archive), 'r')

    # get all years in the hdf5 file (each year is a group)
    years = list(hf.keys())

    if args.start_year is None and args.end_year is None:
        # create a file for each year
        for year in years:
            create_files(args.archive, hf, year)
    else:
        create_files(args.archive, hf, year=None, start_year=args.start_year, end_year=args.end_year)



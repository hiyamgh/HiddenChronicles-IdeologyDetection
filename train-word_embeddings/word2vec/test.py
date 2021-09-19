import os
import h5py
import re
import sys
sys.path.append('..')
from normalization import *

# with open(os.path.join('../some_txt_files/', '33090205.txt'), 'r', encoding='utf-8') as f:
#     print(len(f.readlines()))
# f.close()
#     # print(len(f.read().split('\n')))
#
# with open(os.path.join('../some_txt_files/', '33090205.txt'), 'r', encoding='utf-8') as f:
#     print(len(f.read().split('\n')))
# f.close()


def run():
    # delimiters = EXCLAMATION, en_FULL_STOP, en_SEMICOLON, en_QUESTION, ar_FULL_STOP, ar_SEMICOLON, ar_QUESTION
    delimiters = en_FULL_STOP, ar_FULL_STOP
    # re.escape allows to build the pattern automatically and have the delimiters escaped nicely
    regexPattern = '|'.join(map(re.escape, delimiters))
    # define the Arabic Normalizer instance
    arabnormalizer = ArabicNormalizer()

    year = 1933
    hf = h5py.File('E:/newspapers/nahar.h5', 'r')
    for issue in hf[str(year)].keys():
        unicode_chars = set()
        # print(issue)
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
        for sentence in sentences:
            sentence = sentence.replace('\n', '')
            sentence = sentence.strip()
            if sentence == '':
                continue
            sentence = sentence.split(' ')
            # remove one letter words
            sentence = [s for s in sentence if len(s) > 1]
            for s in sentence:
                if 'u' in s:
                    unicode_chars.add(s)
        if len(unicode_chars):
            print(unicode_chars)


if __name__ == '__main__':
    run()

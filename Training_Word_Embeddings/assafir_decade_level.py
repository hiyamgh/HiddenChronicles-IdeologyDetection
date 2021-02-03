import re
from nltk.stem.isri import ISRIStemmer
from gensim.models import Word2Vec
import glob
from nltk.tokenize import sent_tokenize
import os

MIN_COUNT = 100
SIZE = 300
MIN_YEAR = 1974
MAX_YEAR = 2011


def preprocess_data(text):
    TATWEEL = u"\u0640"
    stemmer = ISRIStemmer()
    text = re.sub('\n', '', text)
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", "", text)
    text = re.sub("«", "", text)
    text = re.sub("\(", "", text)
    text = re.sub(",", "", text)
    text = re.sub("،", "", text)
    text = re.sub("\)", "", text)
    text = re.sub("\(", "", text)
    text = re.sub("»", "", text)
    text = re.sub("«", "", text)
    text = re.sub("؟", "", text)
    text = re.sub("-", "", text)
    text = re.sub(" +", " ", text)

    text = text.lower()
    text = ''.join([i for i in text if not i.isdigit()])  # Remove digits
    text = re.sub(r"http\S+", "", text)  # Remove links
    text = stemmer.norm(text, num=1)  # Remove diacritics
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.replace(TATWEEL, '')

    return text


def get_files_by_year(files, year):
    year_files = []
    for file in files:
        year_suff = file[-12:-10]
        year_suff = int(year_suff)
        if year_suff < 20:
            ystr = '20' + str(year_suff)
        else:
            ystr = '19' + str(year_suff)
        if ystr == year:
            year_files.append(file)

    return year_files


def train_embeddings(batches):
    # combine both batches into 1 list
    assafir_batch1 = glob.glob(batches[0] + "*.txt")
    assafir_batch2 = glob.glob(batches[1] + "*.txt")
    assafir = assafir_batch1 + assafir_batch2

    # loop over all the years for assafir
    for year in range(MIN_YEAR, MAX_YEAR + 1, 10):

        min_year = year
        max_year = year + 10

        data = []
        years = []

        # loop at the decade level (i.e. will create Word Embeddings for each decade )
        for curr_year in range(min_year, max_year):

            # files_year = get_files_by_year(assafir, curr_year)

            for file in assafir:
                # get all files in a certain year
                year_suff = file[-12:-10]
                year_suff = int(year_suff)
                if year_suff < 20:
                    ystr = '20' + str(year_suff)
                else:
                    ystr = '19' + str(year_suff)

                if int(ystr) == year:
                    # print('Processing year %s' % curr_year)
                    if int(ystr) not in years:
                        print('added year: {}'.format(ystr))
                        years.add(int(ystr))

                    file = open(file, "r+", encoding="utf8")

                    # read the text in the file
                    text = file.readlines()
                    text = ' '.join(text)

                    for sentence in sent_tokenize(text):
                        data.append(sentence)

        for i in range(len(data)):
            data[i] = data[i].replace("\n", "")
        for i in range(len(data)):
            data[i] = data[i].split(" ")
        for i in range(len(data)):
            if '' in data[i]:
                data[i].remove('')
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = preprocess_data(data[i][j])

        print('Finished decade %d - %d' % (min_year, max_year))
        print('Years: {}'.format(years))

        modelfinal = Word2Vec(
            data,
            size=SIZE,
            min_count=MIN_COUNT,
            workers=16)
        modelfinal.train(data, total_examples=len(data), epochs=10)
        if not os.path.exists('assafir_models/'):
            os.makedirs('assafir_models/')
        model_name = "assafir_models/cbow_" + str(min_year) + "_" + str(max_year) + ".model"
        modelfinal.save(model_name)


batches = ['E:/newspapers/assafir/assafir/assafir-batch-1/out/',
                'E:/newspapers/assafir/assafir/assafir-batch-2/out/']
train_embeddings(batches)
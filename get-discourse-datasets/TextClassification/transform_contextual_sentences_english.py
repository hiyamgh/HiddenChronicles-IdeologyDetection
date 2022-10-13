import json
import os
import nltk
import pandas as pd
from googletrans import Translator
import numpy as np
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from gramformer import Gramformer
import torch
import en_core_web_sm
nlp = en_core_web_sm.load()


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# def set_seed(seed):
#   torch.manual_seed(seed)
#   if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
#
# set_seed(1212)


def check_corrected(files, input_dir):
    """
     check if all instances have the original sentences (with 'x') and the corrected sentences
    :param files:
    :param input_dir:
    :return:
    """
    corrected = []
    for f in files:
        with open(os.path.join(input_dir, f), encoding="utf8") as json_file:
            data = json.load(json_file)
        for num in data:
            has_x = False
            for lab in data[num]:
                if 'x' in lab:
                    has_x = True
            if has_x:
                pass
            else:
                print('{}: {}'.format(f, num))
            corrected.append(has_x)
    if all(corrected):
        print('All sentences are corrected for their OCR errors')


def transform_ocr_corrected_csv(input_dir, files, dest_dir):
    """ transform the translated - originally OCR corrected - sentences into csv dataset """
    sentences, labels, years = [], [], []
    for f in files:
        with open(os.path.join(input_dir, f), encoding="utf8") as json_file:
            data = json.load(json_file)

            for num in data:
                sentence = ''
                # each num is a sentence
                for label in data[num]:
                    if 'sentence' in label:
                        sentence += " " + data[num][label] + " "
                sentence = sentence.strip()
                label = data[num]["label"]

                sentences.append(sentence)
                labels.append(label)
                years.append(f.split('_')[2][:-5])

    df = pd.DataFrame()
    df['Sentence_ar'] = sentences
    df['Label'] = labels
    df['Year'] = years

    # save the data frame
    mkdir(folder=dest_dir)
    df.to_excel(os.path.join(dest_dir, "sentences_ocr_corrected_discourse_profiling_ar.xlsx"), index=False)
    print('saved data frame into {}'.format(dest_dir))


def transform_ocr_corrected_translated_csv(input_dir, files, dest_dir):
    """ transform the translated - originally OCR corrected - sentences into csv dataset """
    sentences, labels, years = [], [], []
    for f in files:
        with open(os.path.join(input_dir, f), encoding="utf8") as json_file:
            data = json.load(json_file)

            for num in data:
                sentence = ''
                # each num is a sentence
                for label in data[num]:
                    if 'sentence' in label:
                        sentence += " " + data[num][label] + " "
                sentence = sentence.strip()
                label = data[num]["label_old"]

                sentences.append(sentence)
                labels.append(label)
                years.append(f.split('_')[2][:-5])
    df = pd.DataFrame()
    df['Sentence'] = sentences
    df['Label'] = labels
    df['Year'] = years

    # save the data frame
    mkdir(folder=dest_dir)
    df.to_csv(os.path.join(dest_dir, "sentences_ocr_corrected_discourse_profiling_en.csv"), index=False)
    print('saved data frame into {}'.format(dest_dir))


def get_stats(input_dir_ar, input_dir_en, files, discourse_prof_dataset):
    input_dirs = [input_dir_ar, input_dir_en]
    names = ['Arabic directory: {}'.format(input_dir_ar), 'English directory: {}'.format(input_dir_en)]
    all_tokens = []
    for i, input_dir in enumerate(input_dirs):
        print('Processing input directory: {}'.format(names[i]))
        num_tokens = []
        for f in tqdm(os.listdir(input_dir)):
            if f in files:
                print(f)
                with open(os.path.join(input_dir, f), encoding="utf8") as json_file:
                    data = json.load(json_file)

                for num in data:
                    # get the sentences for this number
                    sentence = ''
                    for label in data[num]:
                        if 'sentence' in label and 'x' not in label:
                            sentence += data[num][label]
                    # split the sentence into list of words
                    tokens = sentence.strip().split(" ")
                    num_tokens.append(len(tokens))

        print('Total Number of sentences: {}'.format(len(num_tokens)))
        print('Average number of words per sentence: {:.3f}'.format(np.mean(num_tokens)))
        print('Max number of words per sentence: {:.3f}'.format(max(num_tokens)))
        print('Min number of words per sentence: {:.3f}'.format(min(num_tokens)))

        all_tokens.append(num_tokens)

    print('\nProcessing News Discourse Dataset ...')
    # input/Discourse_Profiling/NewsDiscourse_politicaldiscourse.csv
    df = pd.read_csv(discourse_prof_dataset)
    df_tokens = []
    for i, row in df.iterrows():
        sentence = row['Sentence']
        splitted = sentence.split(" ")
        df_tokens.append(len(splitted))

    print('Total Number of sentences: {}'.format(len(df_tokens)))
    print('Average number of words per sentence: {:.3f}'.format(np.mean(df_tokens)))
    print('Max number of words per sentence: {:.3f}'.format(max(df_tokens)))
    print('Min number of words per sentence: {:.3f}'.format(min(df_tokens)))

    my_dict = {'Sample\nOCR corrected Arabic': all_tokens[0], 'Sample\nOCR Corrected\nTranslated to English': all_tokens[1], 'Discourse\nProfiling\nDataset': df_tokens}

    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    plt.ylabel("Number of tokens per sentence")
    plt.tight_layout()
    plt.savefig('distribution_num_tokens.png', dpi=300)
    plt.close()


def get_distribution_labels(df):
    """
    Draws a stacked bar plot of the distribution of labels over time
    :param df: Data Frame containing columns: Sentence, Label, Year (AFTER OCR correction)
    """
    years = sorted(list(set([int(y) for y in df['Year']])), reverse=True)
    years2labels = {y: [] for y in years}
    df = df.dropna()
    for i, row in df.iterrows():
        y = row['Year']
        label = row['Label']
        years2labels[y].append(label)
    labels = list(set(df['Label']))
    perc = pd.DataFrame()
    for l in labels:
        vals = []
        for year in years:
            val = (len([label for label in years2labels[year] if label == l]) / len(years2labels[year])) * 100
            vals.append(val)
        perc[l] = vals
    perc['Year'] = years
    # plot a Stacked Bar Chart using matplotlib
    perc.plot(
        x='Year',
        kind='barh',
        stacked=True,
        title='Stacked Bar Graph',
        mark_right=True)

    fig = plt.gcf()
    fig.set_size_inches(15, 7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=4, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('distribution_labels.png', dpi=300)
    plt.close()


def transform_corrected_english(input_dir, files, dest_dir, translator, gramformer=None):
    """
    * Transform the OCR-corrected Arabic sentences to English
    * Apply NLTK sentence tokenizer to divide sentences to smaller batches
        because we originally followed the "." in order to divide the text
        to sentences - but since we have OCR errors the "."s are not reliable
        so after correcting OCR and transforming to English we can let NLT do that
        job
    * Save Translated + divided English sentences to JSON file
    :param input_dir: directory of JSON files of sentences + their OCR corrections
    :param files: JSON files of sentences + their OCR corrections
    :param dest_dir: directory to insert JSON files of English translation of sentences
    :param translator: Google Translate instance
    :param gramformer: Grammar Transformer instance
    :return:
    """
    for f in tqdm(files):
        # Open the JSON file of Arabic sentences + their OCR corrections
        with open(os.path.join(input_dir, f), encoding="utf8") as json_file:
            data = json.load(json_file)

        data_new = {}
        count_new = 0

        for num in data:

            # keep them for the record
            keywords_old = data[num]['keywords']
            label_old = data[num]['label']
            year_old = data[num]['year']
            num_old = num

            print('{} - {}'.format(num, f))
            labels = []
            for lab in data[num]:
                if 'x' in lab:
                    labels.append(lab.replace('x', '')) #
            labels = sorted(labels)
            text = ''

            # Noticed that it might be better to
            # translate - NOT IN BATCHES - for the grammar
            for lab in labels:
                text += " " + data[num][lab] + " "

            text = text.replace('\n', '')
            text = text.strip()
            text_en = translator.translate(text, src='ar', dest='en').text
            time.sleep(1)
            for sen in nltk.sent_tokenize(text_en):
                splitline = sen.strip().split(" ")
                data_new[count_new] = {}
                for j in range(0, len(splitline), 10):

                    splitted = splitline[j: j + 10]
                    unsplitted = " ".join(splitted)
                    data_new[count_new]['sentence_{}'.format(j)] = unsplitted

                    # print the english translation in batches
                    print(unsplitted)

                data_new[count_new]["num_old"] = num_old
                data_new[count_new]["keywords"] = keywords_old
                data_new[count_new]["year"] = year_old
                data_new[count_new]["label_old"] = label_old
                data_new[count_new]["label_new"] = ""

                count_new += 1

            splitline_ar = text.strip().split(" ")
            for j in range(0, len(splitline_ar), 10):
                splitted = splitline_ar[j: j + 10]
                unsplitted = " ".join(splitted)
                print(unsplitted)

            print('----------------------------------------------------------------')

        mkdir(folder=dest_dir)
        with open(os.path.join(dest_dir, f), 'w', encoding='utf-8') as fp:
            json.dump(data_new, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("use_gramformer", action="store_true", help="whether to use Gramformer or not")
    args = parser.parse_args()

    # gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

    translator = Translator()
    input_dir = 'sentences/labels_discourse_profiling_ocr_corrected/'
    files = os.listdir(input_dir)
    # files = [f for f in files if f not in ['group_1_1985.json', 'group_1_1986.json', 'group_1_1987.json']]
    dest_dir = 'sentences/labels_discourse_profiling_english/'

    # Translate Arabic - OCR Corrected - sentences to English
    t1 = time.time()
    transform_corrected_english(input_dir=input_dir, files=files, dest_dir=dest_dir, translator=translator, gramformer=None)
    t2 = time.time()
    print('time taken to translate: {} mins'.format((t2-t1)/60))

    # # get statistics about sentences
    discp_path = "input/Discourse_Profiling/NewsDiscourse_politicaldiscourse.csv"
    t1 = time.time()
    get_stats(input_dir_ar=input_dir, input_dir_en=dest_dir, files=files, discourse_prof_dataset=discp_path)
    t2 = time.time()
    print('time taken to get statistics: {} mins'.format((t2 - t1) / 60))

    # transform translated - originally OCR corrected - sentences into a csv file
    transform_ocr_corrected_translated_csv(input_dir=dest_dir, files=files, dest_dir="sentences/csv/")
    # transform OCR corrected (Arabic) - sentences into an excel file
    transform_ocr_corrected_csv(input_dir=input_dir, files=files, dest_dir="sentences/csv/")

    print('English ...')
    df = pd.read_csv("sentences/csv/sentences_ocr_corrected_discourse_profiling_en.csv")
    print('=========================')
    print(df['Label'].value_counts())
    print('=========================')
    print((df['Label'].value_counts()/len(df)) * 100)

    # get the distribution of labels over time (stacked bar plots)
    get_distribution_labels(df=df)

    print('\nArabic ...')
    df = pd.read_excel("sentences/csv/sentences_ocr_corrected_discourse_profiling_ar.xlsx")
    print('=========================')
    print(df['Label'].value_counts())
    print('=========================')
    print((df['Label'].value_counts() / len(df)) * 100)
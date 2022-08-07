import pandas as pd
import os
from sklearn.model_selection import train_test_split
from googletrans import Translator
import time
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def get_class_percentages(df, label_col, mode):
    percentages = (df[label_col].value_counts() / len(df)) * 100
    print('\npercentages in {} data:\n'.format(mode))
    print(percentages)
    print('shape of df: {}'.format(df.shape))


def translate_dataset(df, text_col):
    print('len of df: {}'.format(len(df)))
    df[text_col + '_ar'] = df.apply(lambda x: translator.translate(x[text_col], dest='ar').text, axis=1)
    return df


def clean_data(df, text_col):

    def get_cleaned(text):
        s = re.sub(r'\s*[A-Za-z]+\b', '', text)
        s = re.sub(" \d+", " ", s)
        s = re.sub(r'\b\d+(?:\.\d+)?\s+', '', s)
        s = re.sub('[()]', '', s)  # remove paranthesis
        s = re.sub(r'(?:\-|\s|(\d+))(?=[^><]*?<\/u>)', '', s)
        s = re.sub(r'[^\w]', ' ', s)
        s = re.sub(r'\b(\w+\s*)\1{1,}', '\\1', s) # remove consecutive identical words
        s = re.sub(' +', ' ', s)  # remove multiple spaces
        s = s.strip()
        tokens = [w for w in s.split(' ')]
        s = ' '.join([t for t in tokens if t not in stopwords_list])
        return s

    df[text_col] = df.apply(lambda x: get_cleaned(x[text_col]), axis=1)
    return df


if __name__ == '__main__':
    stopwords_list = stopwords.words('arabic')

    df1 = pd.read_csv('ArgumentationDataset_politicaldiscourse_aljazeera.csv')
    df2 = pd.read_csv('ArgumentationDataset_politicaldiscourse_foxnews.csv')
    df3 = pd.read_csv('ArgumentationDataset_politicaldiscourse_guardian.csv')
    df = pd.concat([df1, df2, df3])

    print(df1.shape, df2.shape, df3.shape, df.shape)

    text_column = 'Sentence'
    label_column = 'Label'
    translator = Translator()

    # remove the class `no-unit` as it is not needed
    df = df[df.Label != 'no-unit']
    df = df[[text_column, label_column]]
    print('shape of df after dropping the `no-unit` class: {}'.format(df.shape))

    # translate the text column into Arabic
    print('\nTranslating {} field into Arabic ...'.format(text_column))
    t1 = time.time()
    df = translate_dataset(df, text_col=text_column)
    t2 = time.time()
    print('time taken to translate: {} mins'.format((t2 - t1) / 60))
    df.to_excel('Argumentation_all_classes_translated.xlsx', index=False)

    # split dataset into 80% training, 10% development, 10% testing
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=list(df[label_column]))
    df_train, df_dev = train_test_split(df_train, test_size=0.1, random_state=42, stratify=list(df_train[label_column]))

    # print class percentages in each dataset
    get_class_percentages(df=df_train, label_col=label_column, mode='train')
    get_class_percentages(df=df_dev, label_col=label_column, mode='dev')
    get_class_percentages(df=df_test, label_col=label_column, mode='test')

    # clean the Arabic translations
    df_train = clean_data(df_train, text_col='Sentence_ar')
    df_dev = clean_data(df_dev, text_col='Sentence_ar')
    df_test = clean_data(df_test, text_col='Sentence_ar')

    df_train.to_excel('df_train.xlsx', index=False, encoding='utf-8-sig')
    df_dev.to_excel('df_dev.xlsx', index=False, encoding='utf-8-sig')
    df_test.to_excel('df_test.xlsx', index=False, encoding='utf-8-sig')
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from googletrans import Translator
import time


def get_class_percentages(df, label_col, mode):
    percentages = (df[label_col].value_counts() / len(df)) * 100
    print('\npercentages in {} data:\n'.format(mode))
    print(percentages)


def translate_dataset(df, text_col):
    print('len of df: {}'.format(len(df)))
    df[text_col + '_ar'] = df.apply(lambda x: translator.translate(x[text_col], dest='ar').text, axis=1)
    return df


if __name__ == '__main__':
    df = pd.read_csv('NewsDiscourse_politicaldiscourse.csv')
    text_column = 'Sentence'
    label_column = 'Label'
    translator = Translator()

    df = df[[text_column, label_column]]
    t1 = time.time()
    df = translate_dataset(df, text_col=text_column)
    t2 = time.time()
    print('time taken to translate: {} mins'.format((t2-t1)/60))

    # split dataset into 80% training, 10% development, 10% testing
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=list(df[label_column]))
    df_train, df_dev = train_test_split(df_train, test_size=0.1, random_state=42, stratify=list(df_train[label_column]))

    # print class percentages in each dataset
    get_class_percentages(df=df_train, label_col=label_column, mode='train')
    get_class_percentages(df=df_dev, label_col=label_column, mode='dev')
    get_class_percentages(df=df_test, label_col=label_column, mode='test')

    df_train.to_excel('df_train.xlsx', index=False, encoding='utf-8-sig')
    df_dev.to_excel('df_dev.xlsx', index=False, encoding='utf-8-sig')
    df_test.to_excel('df_test.xlsx', index=False, encoding='utf-8-sig')


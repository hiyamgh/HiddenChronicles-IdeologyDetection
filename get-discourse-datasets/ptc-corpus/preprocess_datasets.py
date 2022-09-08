import pandas as pd

from dataset import *
import os
from sklearn.model_selection import train_test_split
from googletrans import Translator
import time


def convert_datasets_single_label_multiclass(df_train, df_test, dev_size=0.2, random_state=42):
    df_combined = pd.concat([df_train, df_test])
    # drop all dupliactes (sentences that have similar context are basically
    # sentences with multiple labels - drop all of them)
    df_combined = df_combined.drop_duplicates(subset='context', keep=False) # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
    df_train, df_test = train_test_split(df_combined, test_size=dev_size, random_state=random_state)
    return df_train, df_test

def convert_dataset_multilabel(df):
    labels = ["Appeal_to_fear-prejudice", "Exaggeration,Minimisation", "Repetition", "Doubt",
              "Whataboutism,Straw_Men,Red_Herring", "Black-and-White_Fallacy",
              "Bandwagon,Reductio_ad_hitlerum", "Slogans", "Loaded_Language", "Flag-Waving",
              "Causal_Oversimplification", "Name_Calling,Labeling", "Appeal_to_Authority",
              "Thought-terminating_Cliches"]

    for label in labels:
        df[label] = df.apply(lambda x: 1 if label in x['label'].split(';') else 0, axis=1)
    return df

# def get_class_percentages(df_train, df_dev):
#     percentages_train = (df_train['label'].value_counts() / len(df_train)) * 100
#     percentages_dev = (df_dev['label'].value_counts() / len(df_dev)) * 100
#
#     print('\npercentages in training data:\n')
#     print(percentages_train)
#
#     print('\npercentages in dev data:\n')
#     print(percentages_dev)


def get_class_percentages(df, label_col, mode):
    percentages = (df[label_col].value_counts() / len(df)) * 100
    print('\npercentages in {} data:\n'.format(mode))
    print(percentages)
    print('shape of df: {}'.format(df.shape))


def get_class_counts(df, label_col, mode):
    counts = df[label_col].value_counts()
    print('\ncounts in {} data:\n'.format(mode))
    print(counts)
    print('shape of df: {}'.format(df.shape))


def translate_dataset(df):
    # df = df[:100]
    df = df.iloc[:50]
    print('len of df: {}'.format(len(df)))
    # df['span_ar'] = df.apply(lambda x: translator.translate(x['span'], dest='ar').text, axis=1)
    df['context_ar'] = df.apply(lambda x: translator.translate(x['context'], dest='ar').text, axis=1)
    return df


if __name__ == '__main__':
    train_data_folder = 'train-articles/'
    dev_data_folder = 'dev-articles/'
    labels_path_train = 'train-task-flc-tc.labels'
    labels_path_dev = 'dev-task-flc-tc.labels'
    translator = Translator()

    train_articles, train_ref_articles_id, train_ref_span_starts, train_ref_span_ends, train_labels = load_data(train_data_folder, labels_path_train)
    dev_articles, dev_ref_articles_id, dev_ref_span_starts, dev_ref_span_ends, dev_labels = load_data(dev_data_folder, labels_path_dev)

    t1 = time.time()
    # df_train = dataset_to_pandas(articles=train_articles, ref_articles_id=train_ref_articles_id,
    #                              ref_span_starts=train_ref_span_starts, ref_span_ends=train_ref_span_ends,
    #                              train_gold_labels=train_labels)
    # unique_contexts = list(set(df_train['context']))
    # print(len(unique_contexts)) # 4313
    # print(len(df_train)) # 6128
    #
    # df_train = df_train[['context', 'label']].groupby(['context'])['label'].apply(lambda x: ';'.join(x)).reset_index()
    # df_train = df_train.drop_duplicates()
    # print(len(df_train))

    # print('loaded training dataset ...')
    df_dev = dataset_to_pandas(articles=dev_articles, ref_articles_id=dev_ref_articles_id,
                               ref_span_starts=dev_ref_span_starts, ref_span_ends=dev_ref_span_ends,
                               train_gold_labels=dev_labels)
    unique_contexts = list(set(df_dev['context']))
    print(len(unique_contexts))  # 4313
    print(len(df_dev))  # 6128

    df_dev = df_dev[['context', 'label']].groupby(['context'])['label'].apply(lambda x: ';'.join(x)).reset_index()
    df_dev = df_dev.drop_duplicates()
    print(len(df_dev))
    print('loaded development dataset ...')
    t2 = time.time()
    print('time taken: {} mins'.format((t2-t1)/60))

    df_dev = convert_dataset_multilabel(df=df_dev)

    # df = pd.concat([df_train, df_dev]).reset_index(drop=True)
    # df = df.sample(frac=1).reset_index(drop=True) # shuffle dataframe
    # df_train, df_dev = train_test_split(df, test_size=len(df_dev), random_state=42)
    # t1 = time.time()
    # df_train = translate_dataset(df_train)
    df_dev = translate_dataset(df_dev)

    # df_train.to_excel('df_train_multi.xlsx', index=False, encoding='utf-8-sig')
    df_dev = df_dev.drop(['context', 'label'], axis=1)
    df_dev.to_excel('df_dev_multi.xlsx', index=False, encoding='utf-8-sig')
    for col in df_dev.columns:
        print("\'{}\'".format(col), end=" , ")

    # if os.path.isfile('df_train_single.xlsx'):
    #     df_train = pd.read_excel('df_train_single.xlsx')
    #     df_dev = pd.read_excel('df_dev_single.xlsx')
    #
    #     print('df_train: {}\ndf_dev:{}'.format(df_train.shape, df_dev.shape))
    #     get_class_percentages(df=df_train, label_col='label', mode='train')
    #     get_class_percentages(df=df_dev, label_col='label', mode='validation')
    #
    #     get_class_counts(df=df_train, label_col='label', mode='train')
    #     get_class_counts(df_dev, label_col='label', mode='validation')
    # else:
    #     print(os.getcwd())
    #     translator = Translator()
    #
    #     train_data_folder = 'train-articles/'
    #     dev_data_folder = 'dev-articles/'
    #     labels_path_train = 'train-task-flc-tc.labels'
    #     labels_path_dev = 'dev-task-flc-tc.labels'
    #
    #     train_articles, train_ref_articles_id, train_ref_span_starts, train_ref_span_ends, train_labels = load_data(train_data_folder, labels_path_train)
    #     dev_articles, dev_ref_articles_id, dev_ref_span_starts, dev_ref_span_ends, dev_labels = load_data(dev_data_folder, labels_path_dev)
    #
    #     df_train = dataset_to_pandas(articles=train_articles, ref_articles_id=train_ref_articles_id,
    #                                  ref_span_starts=train_ref_span_starts, ref_span_ends=train_ref_span_ends,
    #                                  train_gold_labels=train_labels)
    #
    #     df_dev = dataset_to_pandas(articles=dev_articles, ref_articles_id=dev_ref_articles_id,
    #                                  ref_span_starts=dev_ref_span_starts, ref_span_ends=dev_ref_span_ends,
    #                                  train_gold_labels=dev_labels)
    #     print('\nBefore transforming to single label / multiclass')
    #     print('df_train: {} / df_test: {}'.format(df_train.shape, df_dev.shape))
    #     df_train, df_dev = convert_datasets_single_label_multiclass(df_train=df_train, df_test=df_dev)
    #     print('\nAfter transforming to single label / multiclass')
    #     print('df_train: {} / df_test: {}'.format(df_train.shape, df_dev.shape))
    #
    #     # print class percentages in each dataset
    #     # get_class_percentages(df_train=df_train, df_dev=df_dev)
    #     get_class_percentages(df=df_train, label_col='label', mode='train')
    #     get_class_percentages(df=df_dev, label_col='label', mode='validation')
    #
    #     get_class_counts(df=df_train, label_col='label', mode='train')
    #     get_class_counts(df_dev, label_col='label', mode='validation')
    #
    #     # since percentages are not equal, will combine both datasets then do a stratified train/test split
    #     df = pd.concat([df_train, df_dev]).reset_index(drop=True)
    #     df = df.sample(frac=1).reset_index(drop=True) # shuffle dataframe
    #
    #     df_train, df_dev = train_test_split(df, test_size=len(df_dev), random_state=42, stratify=df['label'])
    #     print('\nAFTER train/test split:\n')
    #     # get_class_percentages(df_train=df_train, df_dev=df_dev)
    #     get_class_percentages(df=df_train, label_col='label', mode='train')
    #     get_class_percentages(df=df_dev, label_col='label', mode='validation')
    #
    #     get_class_counts(df=df_train, label_col='label', mode='train')
    #     get_class_counts(df_dev, label_col='label', mode='validation')
    #
    #     t1 = time.time()
    #     df_train = translate_dataset(df_train)
    #     df_dev = translate_dataset(df_dev)
    #     t2 = time.time()
    #
    #     print('time taken: {} mins'.format((t2-t1)/60))
    #     df_train.to_excel('df_train_single.xlsx', index=False, encoding='utf-8-sig')
    #     df_dev.to_excel('df_dev_single.xlsx', index=False, encoding='utf-8-sig')
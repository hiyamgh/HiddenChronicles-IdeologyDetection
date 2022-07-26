from dataset import *
import os
from sklearn.model_selection import train_test_split
from googletrans import Translator
import time


def get_class_percentages(df_train, df_dev):
    percentages_train = (df_train['label'].value_counts() / len(df_train)) * 100
    percentages_dev = (df_dev['label'].value_counts() / len(df_dev)) * 100

    print('\npercentages in training data:\n')
    print(percentages_train)

    print('\npercentages in dev data:\n')
    print(percentages_dev)


def translate_dataset(df):
    # df = df[:100]
    print('len of df: {}'.format(len(df)))
    df['span_ar'] = df.apply(lambda x: translator.translate(x['span'], dest='ar').text, axis=1)
    df['context_ar'] = df.apply(lambda x: translator.translate(x['context'], dest='ar').text, axis=1)
    return df


if __name__ == '__main__':
    print(os.getcwd())
    translator = Translator()

    train_data_folder = 'train-articles/'
    dev_data_folder = 'dev-articles/'
    labels_path_train = 'train-task-flc-tc.labels'
    labels_path_dev = 'dev-task-flc-tc.labels'

    train_articles, train_ref_articles_id, train_ref_span_starts, train_ref_span_ends, train_labels = load_data(train_data_folder, labels_path_train)
    dev_articles, dev_ref_articles_id, dev_ref_span_starts, dev_ref_span_ends, dev_labels = load_data(dev_data_folder, labels_path_dev)

    df_train = dataset_to_pandas(articles=train_articles, ref_articles_id=train_ref_articles_id,
                                 ref_span_starts=train_ref_span_starts, ref_span_ends=train_ref_span_ends,
                                 train_gold_labels=train_labels)

    df_dev = dataset_to_pandas(articles=dev_articles, ref_articles_id=dev_ref_articles_id,
                                 ref_span_starts=dev_ref_span_starts, ref_span_ends=dev_ref_span_ends,
                                 train_gold_labels=dev_labels)

    # print class percentages in each dataset
    get_class_percentages(df_train=df_train, df_dev=df_dev)

    # since percentages are not equal, will combine both datasets then do a stratified train/test split
    df = pd.concat([df_train, df_dev]).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle dataframe

    df_train, df_dev = train_test_split(df, test_size=len(df_dev), random_state=42, stratify=df['label'])
    print('\nAFTER train/test split:\n')
    get_class_percentages(df_train=df_train, df_dev=df_dev)

    t1 = time.time()
    df_train = translate_dataset(df_train)
    df_dev = translate_dataset(df_dev)
    t2 = time.time()

    print('time taken: {} mins'.format((t2-t1)/60))
    df_train.to_excel('df_train.xlsx', index=False, encoding='utf-8-sig')
    df_dev.to_excel('df_dev.xlsx', index=False, encoding='utf-8-sig')
import pandas as pd
from googletrans import Translator
import argparse
from tqdm import tqdm
import time
import os
import pickle


def apply_translation(df, args, prefix='train'):
    translator = Translator()
    save_dir = 'translations/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    sentences_lang, labels, labels_general = [], [], []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):

        lang = args.lang[:-1]

        # get english
        sentence_en = row['Sentence']

        try:
            # translate sentence
            sentence_lang = translator.translate(sentence_en, src='en', dest='{}'.format(lang)).text

            # append data
            sentences_lang.append(sentence_lang)

            time.sleep(1)
        except:
            # ignore the error message just for now
            # getting the following in some cases:
            #     raise TypeError(f'the JSON object must be str, bytes or bytearray, '
            # TypeError: the JSON object must be str, bytes or bytearray, not NoneType
            # I guess this has to do with cases where it cannot translate the instance
            # https://github.com/ssut/py-googletrans/issues/301
            continue

        if i % 1000 == 0:
            with open(os.path.join(save_dir, '{}_{}.pickle'.format(prefix, args.lang)), 'wb') as handle:
                pickle.dump(sentences_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, '{}_{}.pickle'.format(prefix, args.lang)), 'wb') as handle:
        pickle.dump(sentences_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="fa", help="language to translate to")
    args = parser.parse_args()

    print('translating from en to {}'.format(args.lang))

    df_train = pd.read_csv('df_train.csv')
    df_val = pd.read_csv('df_validation.csv')
    df_test = pd.read_csv('df_test.csv')

    apply_translation(df=df_train, args=args, prefix='train')
    # apply_translation(df=df_val, args=args, prefix='validation')
    # apply_translation(df=df_test, args=args, prefix='test')


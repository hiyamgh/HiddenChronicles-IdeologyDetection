import time

import pandas as pd
import os
from googletrans import Translator
from tqdm import tqdm
from time import sleep


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# https://py-googletrans.readthedocs.io/en/latest/
# French, fr
# Spanish, es
# German, de
# Greek, el
# Bulgarian, bg
# Russian, ru
# Turkish, tr
# Vietnamese, vi
# Thai, th
# Chinese, zh-cn
# Hindi, hi
# Swahili, sw
# Urdu, ur


if __name__ == '__main__':
    save_dir = 'translations/'
    mkdir(save_dir)

    df_train = pd.read_csv('annotations/train_multiclass.csv')
    df_dev = pd.read_csv('annotations/dev_multiclass.csv')

    translator = Translator()

    languages_codes = ['fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'vi', 'th', 'zh-cn', 'hi', 'sw', 'ur']
    languages_names = ['french', 'spanish', 'german', 'greek', 'bulgarian', 'russian', 'turkish', 'vietnamese',
                       'thai', 'chinese (simplified)', 'hindi', 'swahili', 'urdu']

    for i, lang in enumerate(languages_codes):

        df_trans = pd.DataFrame(columns=['Sentence', 'Label'])

        t1 = time.time()
        for j, row in tqdm(df_train.iterrows(), total=df_train.shape[0], desc='Translating train_multiclass to {}'.format(languages_names[i])):
            sentence = row['context'].strip()
            translated = translator.translate('{}'.format(sentence), src='en', dest=languages_codes[i])
            df_trans = df_trans.append({
                'Sentence': translated,
                'Label': row['label']
            }, ignore_index=True)
            sleep(3)

            if j%100 == 0:
                df_trans.to_excel(os.path.join(save_dir, 'train_multiclass_{}.xlsx'.format(languages_codes[i])))
        t2 = time.time()
        df_trans.to_excel(os.path.join(save_dir, 'train_multiclass_{}.xlsx'.format(languages_codes[i])))
        print('Time taken: {} mins'.format((t2-t1)/60))

        df_trans = pd.DataFrame(columns=['Sentence', 'Label'])

        t1 = time.time()
        for j, row in tqdm(df_dev.iterrows(), total=df_dev.shape[0], desc='Translating dev_multiclass to {}'.format(languages_names[i])):
            sentence = row['context'].strip()
            translated = translator.translate('{}'.format(sentence), src='en', dest=languages_codes[i])
            df_trans = df_trans.append({
                'Sentence': translated,
                'Label': row['label']
            }, ignore_index=True)
            sleep(3)

            if j % 100 == 0:
                df_trans.to_excel(os.path.join(save_dir, 'dev_multiclass_{}.xlsx'.format(languages_codes[i])))
        t2 = time.time()
        df_trans.to_excel(os.path.join(save_dir, 'dev_multiclass_{}.xlsx'.format(languages_codes[i])))
        print('Time taken: {} mins'.format((t2 - t1) / 60))
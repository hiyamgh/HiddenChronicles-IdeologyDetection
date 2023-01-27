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

    df = pd.read_excel('sentences_annotations.xlsx')
    translator = Translator()

    languages_codes = ['fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'vi', 'th', 'zh-cn', 'hi', 'sw', 'ur']
    languages_names = ['french', 'spanish', 'german', 'greek', 'bulgarian', 'russian', 'turkish', 'vietnamese',
                       'thai', 'chinese (simplified)', 'hindi', 'swahili', 'urdu']

    for i, lang in enumerate(languages_codes):

        df_trans = pd.DataFrame(columns=['Sentence', 'Label'])

        for j, row in tqdm(df.iterrows(), total=df.shape[0], desc='Translating to {}'.format(languages_names[i])):
            sentence = row['Sentence'].strip()
            translated = translator.translate('{}'.format(sentence), src='en', dest=languages_codes[i])
            df_trans = df_trans.append({
                'Sentence': translated,
                'Label': row['Label']
            })
            sleep(3)

            if j%100 == 0:
                df_trans.to_excel(os.path.join(save_dir, 'sentences_annotations_{}.xlsx'.format(languages_codes[i])))

        df_trans.to_excel(os.path.join(save_dir, 'sentences_annotations_{}.xlsx'.format(languages_codes[i])))
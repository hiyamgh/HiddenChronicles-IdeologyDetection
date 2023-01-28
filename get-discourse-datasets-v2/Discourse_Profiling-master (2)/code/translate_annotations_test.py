import time
import argparse
import pandas as pd
import os
from googletrans import Translator
from tqdm import tqdm
from time import sleep


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

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

codes2names = {
    'fr': 'french',
    'es': 'spanish',
    'de': 'german',
    'el': 'greek',
    'bg': 'bulgarian',
    'ru': 'russian',
    'tr': 'turkish',
    'ar': 'arabic',
    'vi': 'vietnamese',
    'th': 'thai',
    'zh-cn': 'chinese (simplified)',
    'hi': 'hindi',
    'sw': 'swahili',
    'ur': 'urdu'
}

# languages_codes = ['fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'vi', 'th', 'zh-cn', 'hi', 'sw', 'ur']
# languages_names = ['french', 'spanish', 'german', 'greek', 'bulgarian', 'russian', 'turkish', 'vietnamese',
#                   'thai', 'chinese (simplified)', 'hindi', 'swahili', 'urdu']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--lang', type=str, default='fr', help='code of the language of translation')
    parser.add_argument('--array_id', type=int, default=0, help='SLURM array job id')
    args = parser.parse_args()

    save_dir = 'translationsv2/'
    mkdir(save_dir)

    codes = list(codes2names.keys())
    # get the language from the SLURM ARRAY TASK ID
    if args.array_id < len(codes2names):
        lang = codes[args.array_id]
    elif args.array_id in list(range(len(codes2names), 2*len(codes2names))):
        lang = codes[args.array_id - len(codes2names)]
    else:
        lang = codes[args.array_id - 2*len(codes2names)]

    print('SLURM ARRAY TASK ID: {}, LANGUAGE: {}'.format(args.array_id, lang))

    df = pd.read_csv('df_test.csv')
    print('df.shape before dropping duplicates: {}'.format(df.shape))
    df = df.drop_duplicates()
    print('df.shape after dropping duplicates: {}'.format(df.shape))
    translator = Translator()

    df_trans = pd.DataFrame(columns=['Original', 'Sentence', 'Label', 'Label_general', 'Speech_label'])
    df_to_save_name = 'df_test_{}.xlsx'.format(lang)

    if os.path.isfile(os.path.join(save_dir, df_to_save_name)):
        df_trans = pd.read_excel(os.path.join(save_dir, df_to_save_name))
        num_rows = len(df_trans) # number of rows translated so far
        if num_rows < len(df):
            t1 = time.time()
            for j, row in tqdm(df.iterrows(), total=df.shape[0], desc='Translating to {}'.format(codes2names[lang])):
                if j >= num_rows:
                    sentence = row['Sentence'].strip()
                    if sentence != "":
                        try:
                            translated = translator.translate('{}'.format(sentence), src='en', dest=lang).text
                        except:
                            try:
                                print('was not able to translate row {}: {}'.format(j, row['Sentence']))
                                translator_temp = Translator(service_urls=['translate.googleapis.com'])
                                translated = translator.translate("{}".format(sentence), dest=lang).text
                            except:
                                print('sentence at row {} is set to empty'.format(j))
                                translated = ""
                    else:
                        print('sentence at row {} is empty'.format(j))
                        translated = ""
                    df_trans = df_trans.append({
                        'Original': sentence.strip(),
                        'Sentence': translated,
                        'Label': row['Label'],
                        'Label_general': row['Label_general'],
                        'Speech_label': row['Speech_label']
                    }, ignore_index=True)

                    sleep(1 + args.array_id)

                    if j % 50 == 0:
                        sleep(60)

                    if j % 100 == 0:
                        df_trans.to_excel(os.path.join(save_dir, df_to_save_name), index=False)
            t2 = time.time()
            df_trans.to_excel(os.path.join(save_dir, df_to_save_name), index=False)
            print('Time taken: {} mins'.format((t2 - t1) / 60))
        else:
            print('Completed translating data to {}'.format(lang))
    else:
        t1 = time.time()
        for j, row in tqdm(df.iterrows(), total=df.shape[0], desc='Translating to {}'.format(codes2names[lang])):
            sentence = row['Sentence'].strip()
            if sentence != "":
                try:
                    translated = translator.translate('{}'.format(sentence), src='en', dest=lang).text
                except:
                    try:
                        print('was not able to translate row {}: {}'.format(j, row['Sentence']))
                        translator_temp = Translator(service_urls=['translate.googleapis.com'])
                        translated = translator.translate("{}".format(sentence), dest=lang).text
                    except:
                        print('sentence at row {} is set to empty'.format(j))
                        translated = ""
            else:
                print('sentence at row {} is empty'.format(j))
                translated = ""
            df_trans = df_trans.append({
                'Original': sentence.strip(),
                'Sentence': translated,
                'Label': row['Label'],
                'Label_general': row['Label_general'],
                'Speech_label': row['Speech_label']
            }, ignore_index=True)

            sleep(1 + args.array_id)

            if j % 50 == 0:
                sleep(60)

            if j % 100 == 0:
                df_trans.to_excel(os.path.join(save_dir, df_to_save_name), index=False)
        t2 = time.time()
        df_trans.to_excel(os.path.join(save_dir, df_to_save_name), index=False)
        print('Time taken: {} mins'.format((t2 - t1) / 60))
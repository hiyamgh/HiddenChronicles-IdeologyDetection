import pandas as pd
from sklearn.model_selection import train_test_split
import os
from googletrans import Translator
import time

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':

    save_dir1 = 'split_specific/'
    save_dir2 = 'split_general/'

    # English
    df = pd.read_csv('sentences.csv')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label'])

    mkdir(save_dir1)
    df_train.to_csv(os.path.join(save_dir1, 'sentences_train8.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir1, 'sentences_test8.csv'), index=False)

    df = pd.read_csv('sentences.csv')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label_general'])

    mkdir(save_dir2)
    df_train.to_csv(os.path.join(save_dir2, 'sentences_train3.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir2, 'sentences_test3.csv'), index=False)


    # Arabic
    df = pd.read_excel('sentences.xlsx')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label'])
    df_train.to_excel(os.path.join(save_dir1, 'sentences_train8.xlsx'), index=False)
    df_test.to_excel(os.path.join(save_dir1, 'sentences_test8.xlsx'), index=False)

    df = pd.read_excel('sentences.xlsx')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label_general'])
    df_train.to_excel(os.path.join(save_dir2, 'sentences_train3.xlsx'), index=False)
    df_test.to_excel(os.path.join(save_dir2, 'sentences_test3.xlsx'), index=False)

    # Other langs
    langs = ['de', 'el', 'fa', 'he']
    translator = Translator()
    for lang in langs:
        translations, labels, labels_general = [], [], []
        df_translated = pd.DataFrame()
        for i, row in df.iterrows():
            sentence_ar = row['Sentence']
            label = row['Label']
            label_general = row['Label_general']
            try:
                sentence_lang = translator.translate(sentence_ar, src='ar', dest='{}'.format(lang)).text
                translations.append(sentence_lang)
                labels.append(label)
                labels_general.append(label_general)
                time.sleep(1)
            except:
                continue

        # Save the translations
        df_translated['Sentence'] = translations
        df_translated['Label'] = labels
        df_translated['Label_general'] = labels_general

        df_train, df_test = train_test_split(df_translated, test_size=0.3, random_state=42, stratify=df['Label'])
        df_train.to_excel(os.path.join(save_dir1, 'sentences_{}_train8.xlsx'.format(lang)), index=False)
        df_test.to_excel(os.path.join(save_dir1, 'sentences_{}_test8.xlsx'.format(lang)), index=False)

        df_train, df_test = train_test_split(df_translated, test_size=0.3, random_state=42, stratify=df['Label_general'])
        df_train.to_excel(os.path.join(save_dir2, 'sentences_{}_train3.xlsx'.format(lang)), index=False)
        df_test.to_excel(os.path.join(save_dir2, 'sentences_{}_test3.xlsx').format(lang), index=False)
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':

    save_dir1 = 'csv_annotated/split_specific/'
    save_dir2 = 'csv_annotated/split_general/'

    # stratified split - according to the 8 labels
    # English
    df = pd.read_csv('csv_annotated/sentences.csv')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label'])

    mkdir(save_dir1)
    df_train.to_csv(os.path.join(save_dir1, 'sentences_train8.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir1, 'sentences_test8.csv'), index=False)

    # Arabic
    df = pd.read_excel('csv_annotated/sentences.xlsx')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label'])
    df_train.to_excel(os.path.join(save_dir1, 'sentences_train8.xlsx'), index=False)
    df_test.to_excel(os.path.join(save_dir1, 'sentences_test8.xlsx'), index=False)

    # stratified split - according to 3 labels
    # English
    df = pd.read_csv('csv_annotated/sentences.csv')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label_general'])

    mkdir(save_dir2)
    df_train.to_csv(os.path.join(save_dir2, 'sentences_train3.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir2, 'sentences_test3.csv'), index=False)

    # Arabic
    df = pd.read_excel('csv_annotated/sentences.xlsx')
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label_general'])
    df_train.to_excel(os.path.join(save_dir2, 'sentences_train3.xlsx'), index=False)
    df_test.to_excel(os.path.join(save_dir2, 'sentences_test3.xlsx'), index=False)
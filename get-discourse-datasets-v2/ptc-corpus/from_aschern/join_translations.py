import os
import pandas as pd
import numpy as np


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':

    translation_dir = 'translationsv2/'
    save_dir = 'translations_joined/'
    langs = ['fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'ar', 'vi', 'th', 'zh-cn', 'hi', 'sw', 'ur']

    for lang in langs:
        datasets = []
        for file in os.listdir(translation_dir):
            if '.xlsx' in file and lang in file.split('_')[2]:
                df_temp = pd.read_excel(os.path.join(translation_dir, file))
                datasets.append(df_temp)
        df_joined = pd.concat(datasets)
        assert len(df_joined) == np.sum([len(df) for df in datasets])
        mkdir(save_dir)
        df_joined.to_excel(os.path.join(save_dir, 'df_{}.xlsx'.format(lang)), index=False)

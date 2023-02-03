import os

import pandas as pd

if __name__ == '__main__':
    # annotations = ["VDS", "VDC", "PTC", "ARG"]
    # join the english datasets
    df1 = pd.read_csv('annotations/train_multiclass.csv')
    df2 = pd.read_csv('annotations/dev_multiclass.csv')
    df = pd.concat([df1, df2])
    df.to_excel('annotations/df_multiclass.xlsx', index=False)

    dir = 'translations_joined/'
    path_en = 'annotations/df_multiclass.xlsx'
    added = "../translate_corpora/ptc-corpus/"

    for file in os.listdir(dir):
        if '.xlsx' in file:
            lang = file.split('_')[1][:-5]
            path = os.path.join(dir, file)
            print("\"PTC_{}\": \"{}\",".format(lang, added + path))

    print("\"PTC_en\": \"{}\",".format(added + path_en))
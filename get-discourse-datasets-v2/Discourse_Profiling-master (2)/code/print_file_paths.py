import os

import pandas as pd

if __name__ == '__main__':
    # annotations = ["VDS", "VDC", "PTC", "ARG"]
    # join the english datasets
    df1 = pd.read_csv('df_train.csv')
    df2 = pd.read_csv('df_validation.csv')
    df3 = pd.read_csv('df_test.csv')
    df = pd.concat([df1, df2, df3])
    df.to_excel('df.xlsx', index=False)

    dir = 'translations_joined/'
    path_en = 'df.xlsx'

    for file in os.listdir(dir):
        if '.xlsx' in file:
            lang = file.split('_')[1][:-5]
            path = os.path.join(dir, file)
            added = '../translate_corpora/Discourse_Profiling/'
            print("\"VDC_{}\": \"{}\",".format(lang, added + path))

    print("\"VDC_en\": \"{}\",".format(path_en))
import pandas as pd
import os
import re


def get_political_parties_cleaned(path, df_name):
    df_political_parties = pd.read_csv(os.path.join(path, df_name))
    political_parties = list(df_political_parties['الاسم'])
    pp_cleaned = []
    for pp in political_parties:
        if pp != pp:  # if pp is nan
            continue
        pp = re.sub('\(مترجمه\)', '', pp)
        pp = re.sub('\(لبنان\)', '', pp)
        pp = re.sub('\(توضيح\)', '', pp)
        pp = re.sub('لانغ', '', pp)
        if pp.strip() == '':
            continue
        pp_cleaned.append(pp)
    return list(set(pp_cleaned))


def get_politicians_cleaned(path, df_name):
    df_politicians = pd.read_csv(os.path.join(path, df_name))
    politicians = list(df_politicians['الاسم'])
    p_cleaned = []
    for p in politicians:
        if p != p:  # if pp is nan
            continue
        elif str(p).strip() == '-1':
            continue
        elif str(p).strip() == '1-':
            continue
        p = re.sub('\(مترجمه\)', '', p)
        if p.strip() != '':
            p_cleaned.append(p)
    return list(set(p_cleaned))


if __name__ == '__main__':
    path = 'wikipedia/datasets_updated/'
    # name = 'political_parties_ar.csv'
    # political_parties = get_political_parties_cleaned(path=path, df_name=name)
    # for pp in political_parties:
    #     print(pp)
    name = 'politicians_ar.csv'
    politicians = get_political_parties_cleaned(path=path, df_name=name)
    for p in politicians:
        print(p)
import pandas as pd
import os
import re
import pickle


def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


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


def get_ideologies_cleaned(path, df_name):

    def is_valid(ideol):
        ideol = str(ideol)
        if ideol.strip() == '':
            return False
        if ideol.strip().isdigit():
            return False
        if len(ideol.strip()) == 1:
            return False
        if ideol.strip().replace('-', '').isdigit():  # example: 66-67
            return False
        return True

    df_political_parties = pd.read_csv(os.path.join(path, df_name))
    ideologies = list(df_political_parties['الإيديولوجيا'])
    ideo_cleaned = []
    for ideo in ideologies:
        # if ideo != ideo:  # if pp is nan
        #     continue
        ideo = str(ideo).strip()
        if ideo.isdigit():
            continue
        if len(ideo) <= 1:
            continue
        ideo = re.sub('\(مترجمه\)', '', ideo)
        ideo = re.sub('انظر أدناه', '', ideo)
        ideo = re.sub('الصفحة المطلوبة', '', ideo)
        ideo = re.sub('مطلوب صفحة', '', ideo)
        ideo = re.sub('نفى رسميا', '', ideo)
        ideo = re.sub('156–57', '', ideo)
        ideo = re.sub('151-52', '', ideo)
        ideo = re.sub('465 س', '', ideo)
        ideo = re.sub('156-57 س', '', ideo)
        ideo = re.sub('66-67', '', ideo)
        ideo = re.sub('sfn', '', ideo)
        ideo = re.sub('رفض رسميًا\) قائمة قابلة للطي', '', ideo)
        ideo = re.sub('قائمة بدون تعداد نقطي', '', ideo)
        if ',' in ideo or '،' in ideo or '*' in ideo:
            ideo = ideo.replace('،', ',')
            ideo = ideo.replace('*', ',')
            ideos_sub = ideo.split(',')
            for ids in ideos_sub:
                ids = ids.strip()
                if is_valid(ids):
                    ideo_cleaned.append(ids)
        else:
            if is_valid(ideo):
                # print('ideo is: {}'.format(ideo))
                ideo_cleaned.append(ideo.strip())
            else:
                continue

    return list(set(ideo_cleaned))


def get_politicians_cleaned(path, df_name):
    df_politicians = pd.read_csv(os.path.join(path, df_name))
    politicians = list(df_politicians['الاسم'])
    p_cleaned = []
    for p in politicians:
        p = re.sub('\n', '', p)
        if p != p:  # if pp is nan
            continue
        elif str(p).strip() == '-1':
            continue
        elif str(p).strip() == '1-':
            continue
        elif str(p).strip == '-':
            continue

        p = re.sub('\(مترجمه\)', '', p)

        p = re.sub('\n', '', p)

        if p.strip() != '':
            p_cleaned.append(p)
    return list(set(p_cleaned))


if __name__ == '__main__':
    path = 'wikipedia/datasets_updated/'
    name = 'political_parties_ar.csv'
    save_dir = 'wikipedia/keywords/'
    mkdir(save_dir)
    political_parties = get_political_parties_cleaned(path=path, df_name=name)
    print('political parties')
    for pp in political_parties:
        print(pp)
    with open(os.path.join(save_dir, 'political_parties.pkl'), 'wb') as f:
        pickle.dump(political_parties, f)

    ideologies = get_ideologies_cleaned(path=path, df_name=name)
    print('\nIdeologies')
    for ideo in ideologies:
        print(ideo)
    with open(os.path.join(save_dir, 'ideologies.pkl'), 'wb') as f:
        pickle.dump(ideologies, f)

    name = 'politicians_ar.csv'
    politicians = get_politicians_cleaned(path=path, df_name=name)
    print('\nPoliticians')
    for p in politicians:
        print(p)
    with open(os.path.join(save_dir, 'politicians.pkl'), 'wb') as f:
        pickle.dump(politicians, f)

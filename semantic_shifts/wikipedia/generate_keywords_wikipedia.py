import pandas as pd
import os
import re
import pickle


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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


def clean_political_party(pp):
    pp = str(pp)
    pp = re.sub('\(مترجمه\)', '', pp)
    pp = re.sub('\(لبنان\)', '', pp)
    pp = re.sub('\(توضيح\)', '', pp)
    pp = re.sub('لانغ', '', pp)
    return pp


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


def clean_ideology(ideo):

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

    ideo = str(ideo).strip()
    if ideo.isdigit():
        return ''
    if len(ideo) <= 1:
        return ''
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
        ideo_cleaned = ''
        ideo = ideo.replace('،', ',')
        ideo = ideo.replace('*', ',')
        ideos_sub = ideo.split(',')
        for ids in ideos_sub:
            ids = ids.strip()
            if is_valid(ids):
                ideo_cleaned += ids + ','
        return ideo_cleaned
    return ideo


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


def clean_politician(p):
    p = str(p)
    p = re.sub('\n', '', p)
    if p != p:  # if pp is nan
        return ''
    elif str(p).strip() == '-1':
        return ''
    elif str(p).strip() == '1-':
        return ''
    elif str(p).strip == '-':
        return ''

    p = re.sub('\(مترجمه\)', '', p)
    p = re.sub('\n', '', p)
    if p.strip() != '':
        return p
    return ''


def clean_location_name(l):
    l = str(l)
    if l != l:  # if pp is nan
        return ''
    elif l.strip() == '-1':
        return ''
    elif l.strip() == '1-':
        return ''
    elif l.strip == '-':
        return ''

    l = re.sub('\(مترجمه\)', '', l)
    l = re.sub('\(لبنان\)', '', l)
    l = re.sub('\(توضيح\)', '', l)

    l = re.sub('\n', '', l)
    if l.strip() != '':
        return l
    return ''


if __name__ == '__main__':
    path = 'wikipedia/datasets_updated/'
    name = 'political_parties_ar.csv'
    save_dir = 'keywords/'
    cleaned_dirs = [
        # 'datasets_cleaned/1982-1984/1983_United_States_embassy_bombing_in_Beirut/',
        'datasets_cleaned/1982-1984/1982_Lebanon_War/',
        # 'datasets_cleaned/1982-1984/Bachir_Gemayel_Assasination/',
        # 'datasets_cleaned/1982-1984/May_17_Agreement/',
        # 'datasets_cleaned/1982-1984/Mountain_War_(Lebanon)/',
        # 'datasets_cleaned/1982-1984/Palestinian_insurgency_in_South_Lebanon/',
        # 'datasets_cleaned/1982-1984/Sabra_and_Shatila_massacre/',
        # 'datasets_cleaned/1982-1984/Seige_of_Beirut/',
    ]

    dirs = [
        # 'datasets_updated/1982-1984/1983_United_States_embassy_bombing_in_Beirut/',
        'datasets_updated/1982-1984/1982_Lebanon_War/',
        # 'datasets_updated/1982-1984/Bachir_Gemayel_Assasination/',
        # 'datasets_updated/1982-1984/May_17_Agreement/',
        # 'datasets_updated/1982-1984/Mountain_War_(Lebanon)/',
        # 'datasets_updated/1982-1984/Palestinian_insurgency_in_South_Lebanon/',
        # 'datasets_updated/1982-1984/Sabra_and_Shatila_massacre/',
        # 'datasets_updated/1982-1984/Seige_of_Beirut/',
    ]
    dfnames = [
        'political_parties',
        'politicians',
        'locations',
        'ethnicities_races'
    ]

    for i, dirc in enumerate(dirs):
        for dfname in dfnames:
            if 'Lebanon_War' in dirc:
                print()
            if dfname == 'locations' or dfname == 'ethnicities_races':
                if dfname == 'locations':
                    df_en = pd.read_csv(os.path.join(dirc, dfname + '_en.csv'))
                    df_ar = pd.read_csv(os.path.join(dirc, dfname + '_ar.csv'))

                    # check if the datasets are of same length
                    if len(df_en) != len(df_ar):
                        df_en = df_en.drop(df_en.index[df_en['Name'].str.strip() == '-']).reset_index(drop=True)

                    # drop duplicates
                    df_ar = df_ar.drop(df_en.index[df_en.duplicated(subset=['Name'])]).reset_index(drop=True)
                    df_en = df_en.drop(df_en.index[df_en.duplicated(subset=['Name'])]).reset_index(drop=True)

                    # re-impute missing arabic names
                    en_names = list(df_en['Name'])
                    ar_names = []
                    for j, ar_name in enumerate(df_ar['الاسم']):
                        if ar_name == '-1':
                            name_from_user = input('please write in arabic the following location name: {}'.format(en_names[j]))
                            ar_names.append(name_from_user)
                        else:
                            ar_names.append(ar_name)
                    for n in ar_names:
                        print(n)
                    df_ar['الاسم'] = ar_names
                    df_ar['الاسم'] = df_ar['الاسم'].apply(lambda x: clean_location_name(x))

                    mkdir(cleaned_dirs[i])
                    df_en.to_csv(os.path.join(cleaned_dirs[i], dfname + '_en.csv'), index=False)
                    df_ar.to_csv(os.path.join(cleaned_dirs[i], dfname + '_ar.csv'), index=False, encoding='utf-8-sig')

                else:
                    df = pd.read_csv(os.path.join(dirc, dfname + '.csv'))
                    mkdir(cleaned_dirs[i])
                    df.to_csv(os.path.join(cleaned_dirs[i], dfname + '.csv'), index=False)
            else:
                print(dirc, dfname)
                df_en = pd.read_csv(os.path.join(dirc, dfname + '_en.csv'))
                df_ar = pd.read_csv(os.path.join(dirc, dfname + '_ar.csv'))

                # get entries that are -1 in the english version, delete them from the Arabic version
                df_ar = df_ar.drop(df_en.index[df_en['Name'] == '-']).reset_index(drop=True) # drop the arabic first
                df_en = df_en.drop(df_en.index[df_en['Name'] == '-']).reset_index(drop=True) # then the english

                df_en['Name'] = df_en['Name'].str.strip()
                # drop names that consist of only first name (i.e. string composed of one word)
                idxs2drop = []
                if dfname == 'politicians':
                    for j, val in enumerate(df_en['Name']):
                        if len(str(val).split(' ')) < 2:
                            idxs2drop.append(j)
                    df_ar = df_ar.drop(idxs2drop).reset_index(drop=True)  # drop the arabic first
                    df_en = df_en.drop(idxs2drop).reset_index(drop=True)  # then the english

                df_ar = df_ar.drop(df_en.index[df_en.duplicated(subset=['Name'])]).reset_index(drop=True)
                df_en = df_en.drop(df_en.index[df_en.duplicated(subset=['Name'])]).reset_index(drop=True)

                df_ar = df_ar.drop(df_en.index[df_en['Name'].isnull()]).reset_index(drop=True)
                df_en = df_en.drop(df_en.index[df_en['Name'].isnull()]).reset_index(drop=True)

                if dfname == 'politicians':
                    en_names = list(df_en['Name'])
                    ar_names = []
                    for j, ar_name in enumerate(df_ar['الاسم']):
                        if ar_name == '-1':
                            name_from_user = input('please write in arabic the following name: {}'.format(en_names[j]))
                            ar_names.append(name_from_user)
                        else:
                            ar_names.append(ar_name)
                    for n in ar_names:
                        print(n)
                    df_ar['الاسم'] = ar_names
                    df_ar['الاسم'] = df_ar['الاسم'].apply(lambda x: clean_politician(x))
                    df_ar['الحزب السياسي'] = df_ar['الحزب السياسي'].apply(lambda x: clean_politician(x))

                if dfname == 'political_parties':
                    en_names_pp = list(df_en['Name'])
                    ar_names = []
                    for j, ar_name in enumerate(df_ar['الاسم']):
                        if str(ar_name) == 'nan':
                            name_from_user = input('please write in arabic the following name: {}'.format(en_names_pp[j]))
                            ar_names.append(name_from_user)
                        else:
                            ar_names.append(ar_name)
                    for n in ar_names:
                        print(n)
                    df_ar['الاسم'] = ar_names
                    df_ar['الاسم'] = df_ar['الاسم'].apply(lambda x: clean_politician(x))
                    df_ar['الإيديولوجيا'] = df_ar['الإيديولوجيا'].apply(lambda x: clean_ideology(x))

                mkdir(cleaned_dirs[i])
                df_en.to_csv(os.path.join(cleaned_dirs[i], dfname + '_en.csv'), index=False)
                df_ar.to_csv(os.path.join(cleaned_dirs[i], dfname + '_ar.csv'), index=False, encoding='utf-8-sig')
                print('========================================================================')

    # mkdir(save_dir)
    # political_parties = get_political_parties_cleaned(path=path, df_name=name)
    # print('political parties')
    # for pp in political_parties:
    #     print(pp)
    # with open(os.path.join(save_dir, 'political_parties.pkl'), 'wb') as f:
    #     pickle.dump(political_parties, f)
    #
    # ideologies = get_ideologies_cleaned(path=path, df_name=name)
    # print('\nIdeologies')
    # for ideo in ideologies:
    #     print(ideo)
    # with open(os.path.join(save_dir, 'ideologies.pkl'), 'wb') as f:
    #     pickle.dump(ideologies, f)
    #
    # name = 'politicians_ar.csv'
    # politicians = get_politicians_cleaned(path=path, df_name=name)
    # print('\nPoliticians')
    # for p in politicians:
    #     print(p)
    # with open(os.path.join(save_dir, 'politicians.pkl'), 'wb') as f:
    #     pickle.dump(politicians, f)

import os
import pandas as pd
import argparse


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


parser = argparse.ArgumentParser(description='Creating Testing sets for Discourse Classification')
parser.add_argument('--path', type=str, default='/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/data/', help='path to data folders')
parser.add_argument('--archive', type=str, default='nahar', help='name of the archive')
parser.add_argument('--save_dir', type=str, default='testing_datasets_discourse/', help='directory to save dataset in')
args = parser.parse_args()

if __name__ == '__main__':

    keywords = ['مقاومه', 'اسرائيل', 'حزب+الله', 'سوري']
    keywords_en = ['Mukawama', 'Israel', 'Hezbollah', 'Syrian']
    keywordsar2en = dict(zip(keywords, keywords_en))
    years = [1982, 1985, 1986, 2001, 2002, 2005, 2006, 2007]

    path = os.path.join(args.path, args.archive)
    # path = args.path
    archive = args.archive
    save_dir = os.path.join(args.save_dir, archive)
    # for year in years:
    #     df = pd.DataFrame(columns=['Sentence', 'Year', 'Keyword'])
    #     sentences_sv, years_sv, keywords_sv = [], [], []
    #     for keyword in keywords:
    #         count = 0
    #         if count == 0:
    #             print('processing for keyword {} in {}-{}'.format(keyword, year, archive))
    #         if '+' in keyword:
    #             with open(os.path.join(path, '{}.txt'.format(year)), 'r', encoding='latin-1') as f:
    #                 lines = f.readlines()
    #                 for line in lines:
    #                     tokens = line.strip().split(' ')
    #
    #                     keywords_s = keyword.split('+')
    #                     if all([k in tokens for k in keywords_s]) or all(any([k in t for t in tokens]) for k in keywords_s):
    #                         sentences_sv.append(line)
    #                         years_sv.append(year)
    #                         keywords_sv.append(keyword)
    #                         count += 1
    #
    #             print('total count of {} in {}-{}: {}/{}, which is {}%'.format(keyword, archive, year, count,
    #                                                                                len(lines),
    #                                                                                (count / len(lines)) * 100))
    #             f.close()
    #         else:
    #             with open(os.path.join(path, '{}.txt'.format(year)), 'r', encoding='latin-1') as f:
    #                 lines = f.readlines()
    #                 for line in lines:
    #                     tokens = line.strip().split(' ')
    #                     if keyword in tokens or any([keyword in t for t in tokens]):
    #                         sentences_sv.append(line)
    #                         years_sv.append(year)
    #                         keywords_sv.append(keyword)
    #                         count += 1
    #
    #             print('total count of {} in {}-{}: {}/{}, which is {}%'.format(keyword, archive, year, count, len(lines), (count / len(lines)) * 100))
    #             f.close()
    #
    #     df['Sentence'] = sentences_sv
    #     df['Year'] = years_sv
    #     df['Keyword'] = keywords_sv
    #
    #     mkdir(save_dir)
    #     df.to_excel(os.path.join(save_dir, 'df_test_{}.xlsx'.format(year)), index=False)

    for file in ['df_test_{}.xlsx'.format(y) for y in [1982, 1985, 1986, 2001, 2002, 2005, 2006, 2007]]:
        df = pd.read_excel(os.path.join(save_dir, file))
        keywords_df_en = []
        for i, row in df.iterrows():
            k_ar = row['Keyword']
            k_en = keywordsar2en[k_ar]
            keywords_df_en.append(k_en)

        df['Keyword_en'] = keywords_df_en
        print('{}: {}'.format(file, list(set(list(df['Keyword'])))))
        for i, keyword in enumerate(keywords):
            keyword_en = keywordsar2en[keyword]
            df_sub = df[df['Keyword_en'] == keyword_en]
            if len(df_sub) > 1:
                df_sub.to_excel(os.path.join(save_dir, '{}_{}.xlsx'.format(file[:-5], keywords_en[i])), index=False)
            else:
                print('word {}/{} not found in {}'.format(keyword, keyword_en, file))
        print('===================================================================================')

    for file in ['df_test_{}_Mukawama.xlsx'.format(y) for y in [1982, 1985, 1986, 2001, 2002, 2005, 2006, 2007]]:
        df = pd.read_excel(os.path.join(save_dir, file))
        print('{} shape: {}'.format(file, df.shape))

    for file in ['df_test_{}_Israel.xlsx'.format(y) for y in [1982, 1985, 1986, 2001, 2002, 2005, 2006, 2007]]:
        if os.path.isfile(os.path.join(save_dir, file)):
            df = pd.read_excel(os.path.join(save_dir, file))
            print('{} shape: {}'.format(file, df.shape))
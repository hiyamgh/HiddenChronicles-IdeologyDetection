import os
import pandas as pd
import argparse


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


parser = argparse.ArgumentParser(description='Creating Testing sets for Discourse Classification')
parser.add_argument('--path', type=str, default='/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/data/', help='path to data folders')
parser.add_argument('--archive', type=str, default='nahar', help='name of the archive')
parser.add_argument('--save_dir', type=str, default='/scratch/7613491_hkg02/political_discourse_mining_hiyam/get-discourse-datasets/testing_datasets_political_discourse/', help='directory to save dataset in')
args = parser.parse_args()

if __name__ == '__main__':

    years2keywords = {
        '1982-1984': ['اسرائيل', 'ياسر+عرفات', 'منظمة+التحرير+الفلسطينية', 'بشير+الجميل', 'صبرا+شاتيلا', 'السفاره+الامريكيه'],
        '2005-2006': ['الحريري'],
        '2006-2009': ['اسرائيل', 'سلاح+مقاومه']
    }

    path = os.path.join(args.path, args.archive)
    archive = args.archive
    save_dir = args.save_dir
    df = pd.DataFrame(columns=['Sentence', 'Year', 'Keyword'])
    sentences_sv, years_sv, keywords_sv = [], [], []
    for years in years2keywords:
        years_s = years.split('-')
        for year in range(int(years_s[0]), int(years_s[1]) + 1):
            for keyword in years2keywords[years]:
                if '+' in keyword:
                    print('processing for keyword {} in {}-{}'.format(keyword, year, archive))
                    count = 0

                    with open(os.path.join(path, '{}.txt'.format(year)), 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            tokens = line.strip().split(' ')

                            keywords_s = keyword.split('+')
                            if all([k in tokens for k in keywords_s]) or all(any([k in t for t in tokens]) for k in keywords_s):
                                sentences_sv.append(line)
                                years_sv.append(year)
                                keywords_sv.append(keyword)
                                count += 1

                        print('total count of {} in {}-{}: {}/{}, which is {}%'.format(keyword, archive, year, count, len(lines), (count / len(lines)) * 100))
                    f.close()
                else:
                    print('processing for keyword {} in {}-{}'.format(keyword, year, archive))
                    count = 0
                    with open(os.path.join(path, '{}.txt'.format(year)), 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            tokens = line.strip().split(' ')
                            if keyword in tokens or any([keyword in t for t in tokens]):
                                sentences_sv.append(line)
                                years_sv.append(year)
                                keywords_sv.append(keyword)
                                count += 1

                        print('total count of {} in {}-{}: {}/{}, which is {}%'.format(keyword, archive, year, count, (lines), (count / len(lines)) * 100))
                    f.close()

    df['Sentence'] = sentences_sv
    df['Year'] = years_sv
    df['Keyword'] = keywords_sv

    mkdir(args.save_dir)
    df.to_excel(os.path.join(save_dir, 'df_test_{}.xlsx'.format(archive)), index=False)
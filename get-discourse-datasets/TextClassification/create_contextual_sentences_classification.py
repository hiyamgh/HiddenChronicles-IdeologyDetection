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
    years = [1982, 1985, 1986, 2001, 2002, 2005, 2006, 2007]

    path = os.path.join(args.path, args.archive)
    archive = args.archive
    save_dir = args.save_dir
    for year in years:
        df = pd.DataFrame(columns=['Sentence', 'Year', 'Keyword'])
        sentences_sv, years_sv, keywords_sv = [], [], []
        for keyword in keywords:
            if '+' in keyword:
                print('processing for keyword {} in {}-{}'.format(keyword, year, archive))
                count = 0

                with open(os.path.join(path, '{}.txt'.format(year)), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        tokens = line.strip().split(' ')

                        keywords_s = keyword.split('+')
                        if all([k in tokens for k in keywords_s]) or all(
                                any([k in t for t in tokens]) for k in keywords_s):
                            sentences_sv.append(line)
                            years_sv.append(year)
                            keywords_sv.append(keyword)
                            count += 1

                    print('total count of {} in {}-{}: {}/{}, which is {}%'.format(keyword, archive, year, count,
                                                                                   len(lines),
                                                                                   (count / len(lines)) * 100))
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

                    print(
                        'total count of {} in {}-{}: {}/{}, which is {}%'.format(keyword, archive, year, count, (lines),
                                                                                 (count / len(lines)) * 100))
                f.close()

        df['Sentence'] = sentences_sv
        df['Year'] = years_sv
        df['Keyword'] = keywords_sv

        mkdir(args.save_dir)
        df.to_excel(os.path.join(save_dir, 'df_test_{}_{}.xlsx'.format(archive, year)), index=False)
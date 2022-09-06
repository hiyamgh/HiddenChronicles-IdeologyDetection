import os
import random
import pandas as pd
import json

path = 'C:/Users/96171/Downloads/'
files = ['1982.txt', '1983.txt', '1984.txt', '1985.txt', '1986.txt', '1987.txt']
random.seed(42)

for file in files:
    samples = {}
    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
        lines = f.readlines()[:10]
        sample = random.sample(lines, k=3)

        for i, line in enumerate(sample):
            splitline = line.split(' ')
            batch = ''
            for j in range(0, len(splitline), 10):
                splitted = splitline[j: j + 10]
                batch = ' '.join(splitted)
                if i not in samples:
                    samples[i] = {}
                samples[i]['sentence_{}'.format(j)] = batch
            samples[i]['label'] = ''
        with open('{}.json'.format(file[:-4]), 'w', encoding='utf-8') as fp:
            json.dump(samples, fp, indent=4, ensure_ascii=False)
    print('sample lines written to {}_forlabelling.txt'.format(file[:-4]))
    f.close()

# for file in files:
#     with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         sample = random.sample(lines, k=300)
#
#         with open(os.path.join(path, '{}_sample.txt'.format(file[:-4])), 'w', encoding='utf-8') as fsample:
#             for line in sample:
#                 fsample.write(line)
#         fsample.close()
#     print('sample lines written to {}_sample.txt'.format(file[:-4]))
#     f.close()
#
# path = 'C:/Users/96171/Downloads/'
# df = pd.read_csv('input/FAKES/feature_extraction_train_updated_updated.csv')
# with open(os.path.join(path, 'train_s1.txt'), 'w', encoding='utf-8') as f:
#     for i, row in df.iterrows():
#         article = row['article_title']
#         f.write(article+'\n')
# f.close()
#
# df = pd.read_csv('input/FAKES/feature_extraction_dev_updated_updated.csv')
# path = 'C:/Users/96171/Downloads/'
# with open(os.path.join(path, 'train_s2.txt'), 'w', encoding='utf-8') as f:
#     for i, row in df.iterrows():
#         article = row['article_title']
#         f.write(article+'\n')
# f.close()
#
# df = pd.read_csv('input/FAKES/feature_extraction_test_updated.csv')
# path = 'C:/Users/96171/Downloads/'
# with open(os.path.join(path, 'test_s.txt'), 'w', encoding='utf-8') as f:
#     for i, row in df.iterrows():
#         article = row['article_title']
#         f.write(article+'\n')
# f.close()
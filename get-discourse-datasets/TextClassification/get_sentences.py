import os
import pandas as pd
import random
import argparse
import json

# 1982 - 1987: -----------------------------------------------------------
# 1982 Seige of Beirut: حصار بيروت
# 1982 Palestinian_insurgency_in_South_Lebanon: المقاومة_الفلسطينية_في_جنوب_لبنان
# 1982 Sabra and Shatila massacre: مجزرة صبرا وشاتيلا
# 1983 United States Embassy Bombing in Beirut: تفجير السفارة الأمريكية في بيروت
# 3 September 1983 – February 1984: Mountain War: حرب الجبل
# 19 May 1985 – July 1988: War of the Camps: حرب المخيمات

# https://towardsdatascience.com/fine-tuning-for-domain-adaptation-in-nlp-c47def356fd6
# https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
#
# NB-MLM: Efficient Domain Adaptation of Masked Language Models for
# Sentiment Analysis --> https://github.com/SamsungLabs/NB-MLM
#
# check out google scholar: https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=fine+tune+bert+domain+adaptation+mlm&btnG=


def samples2json(keywords, years, sentences):
    samples = {}
    for i, line in enumerate(sentences):
        splitline = line.split(' ')
        for j in range(0, len(splitline), 10):
            splitted = splitline[j: j + 10]
            batch = ' '.join(splitted)
            if i not in samples:
                samples[i] = {}
            samples[i]['sentence_{}'.format(j)] = batch
        samples[i]['keywords'] = keywords[i]
        samples[i]['year'] = years[i]
        samples[i]['label'] = ''
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--archive", default="nahar", type=str, help="The archive to get sentences from.")
    parser.add_argument("--save_dir", default="sentences/", type=str, help="directory to save the output files in")
    args = parser.parse_args()

    groups = ['مقاومه,فلسطينيه,جنوب,لبنان', 'مجزره,صبرا,شاتيلا', 'تفجير,السفاره,الامريكيه,بيروت']
    group_names = ['group_{}'.format(i) for i in range(len(groups))]
    directory = '/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/data/{}/'.format(args.archive)
    # directory = 'C:/Users/96171/Downloads/'
    files = ['1982.txt', '1983.txt', '1984.txt', '1985.txt', '1986.txt', '1987.txt']
    save_dir = 'sentences/{}/'.format(args.archive)

    print('processing sentences from {} archive ...\n'.format(args.archive))

    for i, group in enumerate(groups):
        for file_name in files:
            if os.path.isfile(os.path.join(directory, file_name)):
                df = pd.DataFrame(columns=['keywords', 'year', 'sentence'])
                with open(os.path.join(directory, file_name), 'r', encoding='utf-8-sig') as f:
                    lines = f.readlines()
                f.close()
                print('file: {}, lines: {}'.format(file_name, len(lines)))
                keywords, years, sentences = [], [], []
                words = group.split(',')
                for line in lines:
                    tokens = line.split(' ')
                    match = True
                    for w in words:
                        if (w in tokens) or any([w in t for t in tokens]):
                            pass
                        else:
                            match = False
                            break
                    if match:
                        keywords.append(group)
                        years.append(file_name[:-4])
                        sentences.append(line)

                print('file: {}, group: {}'.format(file_name, group))
                print(len(keywords), len(sentences), len(years), '{:.3f}%'.format((len(sentences) / len(lines)) * 100))
                if len(sentences) > 0:
                    print('average number of tokens per sentence: {}'.format(
                        sum([len(line.strip().split(' ')) for line in sentences]) / len(sentences)))

                idxs_normal = [i for i in range(len(sentences)) if len(sentences[i]) <= 512]
                sentences_n = [sentences[i] for i in idxs_normal]
                keywords_n = [keywords[i] for i in idxs_normal]
                years_n = [years[i] for i in idxs_normal]

                random.seed(42)
                if len(sentences_n) < 50:
                    random_idxs = [i for i in range(len(sentences_n))]  # just get them all
                else:
                    random_idxs = random.sample([i for i in range(len(sentences_n))], k=50)
                rand_keywords, rand_years, rand_sentences = [], [], []
                for ridx in random_idxs:
                    rand_keywords.append(keywords_n[ridx])
                    rand_years.append(years_n[ridx])
                    rand_sentences.append(sentences_n[ridx])

                samples = samples2json(keywords=rand_keywords, years=rand_years, sentences=rand_sentences)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, '{}_{}.json'.format(group_names[i], file_name[:-4])), 'w', encoding='utf-8') as fp:
                    json.dump(samples, fp, indent=4, ensure_ascii=False)

                print('---------------------------------------')
            else:
                print('No such file or directory: {}'.format(os.path.join(directory, file_name)))
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


# Bachir Gemayel Assassinated in <<1982-1987>>
# https://en.wikipedia.org/wiki/Bachir_Gemayel#Assassination
# بشير,جميل,اغتيال

# The American Embassy in Beirut is attacked by a suicide bomb: <<1983->1987>>
# https://en.wikipedia.org/wiki/1983_United_States_embassy_bombing_in_Beirut
# تفجير,السفاره,الامريكيه,بيروت

# War of Liberation against Syrians
# https://en.wikipedia.org/wiki/War_of_Liberation_(1989%E2%80%931990) <<1989-1990->1993>>
# حرب,تحرير,جنرال,عون

# Oslo agreements <<1993-1996 1993->1996>>
# https://en.wikipedia.org/wiki/Oslo_Accords#:~:text=The%20Oslo%20Accords%20are%20a,Taba%2C%20Egypt%2C%20in%201995.
# اتفاقيات,أوسلو,اسرائيل,منظمه,تحرير,فلسطينيه

# September 11 attacks <<2001-2003>>
# https://en.wikipedia.org/wiki/September_11_attacks
# هجمات,11,(سبتمبر,ايلول),قاعده

# Hariri Assassination <<2005-2006>>
# https://en.wikipedia.org/wiki/Assassination_of_Rafic_Hariri
# اغتيال,رفيق,الحريري

# 2006 Lebanon Israel War <<2006-2008>>
# https://en.wikipedia.org/wiki/2006_Lebanon_War
# حرب,تموز,اسرائيل


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

    groups = [
        ['Mukawama', 'مقاومه,فلسطينيه,جنوب,لبنان', (1982, 1987)],
        ['Sabra_Shatila',  'مجزره,صبرا,شاتيلا', (1982, 1987)],
        ['US_Embassy_Bombing',  'تفجير,سفاره,اميركيه,بيروت', (1983, 1987)],
        ['War_of_Liberation', 'حرب,تحرير', (1989, 1993)],
        ['Oslo_Accords', 'أوسلو,اسرائيل', (1993, 1996)],
        ['September_11_attacks', 'هجمات,ايلول,قاعده', (2001, 2003)],
        ['Rafik_Hariri_Assassination',  'اغتيال,رفيق,الحريري', (2005, 2009)],
        ['Lebanon_Israel_War_2006', 'حرب,تموز,اسرائيل', (2006, 2008)]
    ]

    directory = '/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/data/{}/'.format(args.archive)
    # directory = 'C:/Users/96171/Downloads/'
    print('processing sentences from {} archive ...\n'.format(args.archive))

    for i, group in enumerate(groups):
        group_name, words_grouped, years_range = group[0], group[1], group[2]
        save_dir = 'sentences/{}_{}/'.format(args.archive, group_name)
        years = [y for y in range(years_range[0], years_range[1] + 1)]
        for y in years:
            file_name = '{}.txt'.format(y)
            if os.path.isfile(os.path.join(directory, file_name)):
                with open(os.path.join(directory, file_name), 'r', encoding='utf-8-sig') as f:
                    lines = f.readlines()
                f.close()
                print('file: {}, lines: {}'.format(file_name, len(lines)))
                keywords, years, sentences = [], [], []
                words = words_grouped.split(',')
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
                        keywords.append(words_grouped)
                        years.append(y)
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
                with open(os.path.join(save_dir, '{}_{}.json'.format(group_name, y)), 'w', encoding='utf-8') as fp:
                    json.dump(samples, fp, indent=4, ensure_ascii=False)

                print('---------------------------------------')
            else:
                print('No such file or directory: {}'.format(os.path.join(directory, file_name)))

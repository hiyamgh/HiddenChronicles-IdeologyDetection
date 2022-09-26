import json
import os
import nltk
from googletrans import Translator
import numpy as np

from gramformer import Gramformer
import torch
# import spacy
# spacy.load('en')

import en_core_web_sm
nlp = en_core_web_sm.load()


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)


gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

# influent_sentences = [
#     "He are moving here.",
#     "I am doing fine. How is you?",
#     "How is they?",
#     "Matt like fish",
#     "the collection of letters was original used by the ancient Romans",
#     "We enjoys horror movies",
#     "Anna and Mike is going skiing",
#     "I walk to the store and I bought milk",
#     " We all eat the fish and then made dessert",
#     "I will eat fish for dinner and drink milk",
#     "what be the reason for everyone leave the company",
# ]
#
# for influent_sentence in influent_sentences:
#     corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
#     print("[Input] ", influent_sentence)
#     for corrected_sentence in corrected_sentences:
#       print("[Correction] ",corrected_sentence)
#     print("-" *100)

translator = Translator()
files = os.listdir('sentences/labels_discourse_profiling_ocr_corrected/')
input_dir = 'sentences/labels_discourse_profiling_ocr_corrected/'
files = [f for f in files if f not in ['group_1_1986.json', 'group_1_1987.json']]
dest_dir = 'sentences/labels_discourse_profiling_english/'

# # check if all instances have the original sentences (with 'x') and the corrected sentences
# for f in files:
#     with open(os.path.join(input_dir, f), encoding="utf8") as json_file:
#         data = json.load(json_file)
#     for num in data:
#         has_x = False
#         for lab in data[num]:
#             if 'x' in lab:
#                 has_x = True
#         if has_x:
#             pass
#         else:
#             print('{}: {}'.format(f, num))

# check if all instances have the original sentences (with 'x') and the corrected sentenecs
sent_lens = []
sent_lens_old = []
count_old = 0
data_new = {}
for f in files:
    with open(os.path.join(input_dir, f), encoding="utf8") as json_file:
        data = json.load(json_file)

    data_new = {}
    count_new = 0

    for num in data:

        # keep them for the record
        keywords_old = data[num]['keywords']
        label_old = data[num]['label']
        year_old = data[num]['year']
        num_old = num

        print('{} - {}'.format(num, f))
        labels = []
        for lab in data[num]:
            if 'x' in lab:
                labels.append(lab.replace('x', ''))
        labels = sorted(labels)
        text = ''
        text_ar = ''

        for lab in labels:
            text_en = translator.translate(data[num][lab], src='ar', dest='en').text
            text_en_corr = gf.correct(text_en, max_candidates=1)
            text += next(iter(text_en_corr)) + " " # the set contains only 1 element so no worries
            text_ar += data[num][lab] + " "
            print('[Arabic]', data[num][lab])
            print('[English]', text_en)
            print('[English Grammar]', text_en_corr)
            print('????????????????????????????????????????????????????????????????')

        count_old += 1

        text = text.replace('\n', '')
        for sen in nltk.sent_tokenize(text):
            splitline = sen.strip().split()
            data_new[count_new] = {}
            for j in range(0, len(splitline), 10):
                splitted = splitline[j: j + 10]
                unsplitted = " ".join(splitted)
                data_new[count_new]['sentence_{}'.format(j)] = unsplitted

            data_new[count_new]["num_old"] = num_old
            data_new[count_new]["keywords"] = keywords_old
            data_new[count_new]["year"] = year_old
            data_new[count_new]["label_new"] = ""

            count_new += 1

            #  statistics
            print(sen)
            sl = len(splitline)
            print(sl)
            sent_lens.append(sl)
            print('-------------------------------------------------------')

        sl_ar = len(text_ar.strip().split(' '))
        sent_lens_old.append(sl_ar)

    mkdir(folder=dest_dir)
    with open(os.path.join(dest_dir, f), 'w', encoding='utf-8') as fp:
        json.dump(data_new, fp, indent=4, ensure_ascii=False)
    break

print('Avg number of words per sentence - original: {}'.format(np.sum(sent_lens_old)/len(sent_lens_old)))
print('Min nb of words per sentence - original: {}'.format(min(sent_lens_old)))
print('Max nb of words per sentence - original: {}'.format(max(sent_lens_old)))

print('\nAvg number of words per sentence - new: {}'.format(np.sum(sent_lens)/len(sent_lens)))
print('Min nb of words per sentence - new: {}'.format(min(sent_lens)))
print('Max nb of words per sentence - new: {}'.format(max(sent_lens)))

# print('\nNumber of sentences - original: {}'.format(count_old))
# print('Number of sentences - new: {}'.format(count_new))
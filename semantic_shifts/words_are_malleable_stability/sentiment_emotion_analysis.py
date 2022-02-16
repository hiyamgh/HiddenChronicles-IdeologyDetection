import pickle
from deep_translator import GoogleTranslator
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.corpus import wordnet
import nltk
from nltk.stem import WordNetLemmatizer
from lang_trans.arabic import buckwalter

print(buckwalter.transliterate(u'سعاده'))
print(buckwalter.untransliterate('$AHinap_1'))
print(buckwalter.untransliterate('>asiyawiy~_1'))
print(buckwalter.untransliterate('qASid_2'))
print(buckwalter.untransliterate('<izoEAj'))

# from camel_tools.utils.charmap import CharMapper
# sentence = "ازعاج"
# print(sentence)
# ar2bw = CharMapper.builtin_mapper('ar2bw')
# sent_bw = ar2bw(sentence)
# print(sent_bw)

# # # create all_summaries dictionary but for English
# # all_summaries_en = {}
# #
# # with open('all_summaries.pkl', 'rb') as handle:
# #     all_summaries = pickle.load(handle)
# # for w in all_summaries:
# #     print(w)
# #     all_summaries_en[w] = {}
# #     for year in all_summaries[w]:
# #         all_summaries_en[w][year] = []
# #         print('YEAR: {}, w: {} ======================================================================================================'.format(year, w))
# #         for neigh in all_summaries[w][year]:
# #             if all_summaries[w][year][neigh] == []:
# #                 print('skipping {} since it was not known in Arabic, so can\'t translate to english'.format(w))
# #             else:
# #                 print(neigh)
# #                 print(all_summaries[w][year][neigh])
# #                 trans = GoogleTranslator(source='ar', target='en').translate(all_summaries[w][year][neigh][0])
# #                 all_summaries_en[w][year].append(trans)
# #                 print(trans)
# #                 print('---------------------------------------- ')
# #
# #         with open('all_summaries_en.pickle', 'wb') as handle:
# #             pickle.dump(all_summaries_en, handle, protocol=pickle.HIGHEST_PROTOCOL)
# #
# # # load the english dictionary
# # with open('all_summaries_en.pkl', 'rb') as handle:
# #     all_summaries_en = pickle.load(handle)
#
# emotions = ['AFRAID', 'AMUSED', 'ANGRY', 'ANNOYED', 'DONT_CARE', 'HAPPY', 'INSPIRED', 'SAD']
# idxs = list(range(len(emotions)))
# # map each emotion to its index
# em2idx = dict(zip(emotions, idxs))
#
# # path = 'C:/Users/96171/Downloads/DepecheMood_english_lemma_full.tsv'
# # all_emotions = {}
# # t1 = time.time()
# # with open(path) as fd:
# #     rd = csv.reader(fd, delimiter="\t")
# #     count = 0
# #     for row in rd:
# #         print(row)
# #         if count != 0:
# #             all_emotions[row[0]] = row[1:-1]
# #             print(sum([float(i) for i in row[1:-1]]))
# #         count += 1
# # t2 = time.time()
# #
# # print('total num: {}'.format(count))
# # print('time taken: {} mins'.format((t2-t1)/60))
# #
# # with open('all_emotions.pkl', 'wb') as f:
# #     pickle.dump(all_emotions, f)
#
# with open('all_emotions.pkl', 'rb') as f:
#     all_emotions = pickle.load(f)
#
# with open('all_summaries_en.pickle', 'rb') as f:
#     all_summaries = pickle.load(f)
# #
# #
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # def get_wordnet_pos(treebank_tag):
# #
# #     if treebank_tag.startswith('J'):
# #         return wordnet.ADJ
# #     elif treebank_tag.startswith('V'):
# #         return wordnet.VERB
# #     elif treebank_tag.startswith('N'):
# #         return wordnet.NOUN
# #     elif treebank_tag.startswith('R'):
# #         return wordnet.ADV
# #     else:
# #         pass
# #
# #
# # all_scores = {}
# # positive_words = []
# # negative_words = []
# #
# # wordnet_lemmatizer = WordNetLemmatizer()
# # for w in all_summaries:
# #     all_scores[w] = {}
# #     for year in all_summaries[w]:
# #         all_scores[w][year] = {}
# #
# #         summary = all_summaries[w][year]
# #
# #         all_scores[w][year]['pos'] = []
# #         all_scores[w][year]['neg'] = []
# #         all_scores[w][year]['obj'] = []
# #
# #         for s in summary:
# #
# #             s = wordnet_lemmatizer.lemmatize(s)
# #             if ' ' in s:
# #                 s_tkn = s.split(' ')
# #             else:
# #                 s_tkn = [s]
# #
# #             for p in nltk.pos_tag(s_tkn):
# #                 get_pos_tag = get_wordnet_pos(p[1])
# #                 # if type(get_pos_tag) == str:
# #                 if isinstance(get_pos_tag, str):
# #
# #                     try:
# #                         synset = swn.senti_synset(p[0] + '.' + get_pos_tag + '.01')
# #
# #                         all_scores[w][year]['pos'].append(synset.pos_score())
# #                         all_scores[w][year]['neg'].append(synset.neg_score())
# #                         all_scores[w][year]['obj'].append(synset.obj_score())
# #
# #                     except Exception as e:
# #                         print(str(e))
# #
# # print(all_scores)
# # for w in all_scores:
# #     for year in all_scores[w]:
# #         print('w: ' + w)
# #         print('year: ' + year)
# #         print('pos: ' + str(np.mean(np.array(all_scores[w][year]['pos'])).astype(np.float)))
# #         print('neg: ' + str(np.mean(np.array(all_scores[w][year]['neg'])).astype(np.float)))
# #         print('obj: ' + str(np.mean(np.array(all_scores[w][year]['obj'])).astype(np.float)))
# #
# #             # Exception as e:  # work on python 3.x
# #             # logger.error('Failed to upload to ftp: ' + str(e))
# #             # print()
#
#
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# all_scores = {}
# for w in all_summaries:
#     all_scores[w] = {}
#     for year in all_summaries[w]:
#         all_scores[w][year] = {}
#         print('w: ' + w)
#         print('year: ' + year)
#         summary = all_summaries[w][year]
#         score = {}
#         for s in summary:
#             if ' ' in s:
#                 splitted = s.split(' ')
#                 canpass = False
#                 for sp in splitted:
#                     if sp in all_emotions:
#                         canpass = True
#                         break
#                 if canpass:
#                     pass
#                 else:
#                     print('no emotions found for {}'.format(s))
#                     continue
#             else:
#                 splitted = s
#                 if splitted in all_emotions:
#                     pass
#                 else:
#                     print('no emotions found for {}'.format(s))
#                     continue
#
#             for emotion in em2idx:
#                 if emotion not in score:
#                     score[emotion] = []
#                 em_idx = em2idx[emotion]
#                 if ' ' in s:
#                     wl = s.split(' ')
#                     for wll in wl:
#                         if wll in all_emotions:
#                             score[emotion].append(all_emotions[wll][em_idx])
#                         else:
#                             print('no emotions found for {}'.format(wll))
#                 else:
#                     score[emotion].append(all_emotions[s][em_idx])
#
#         for k in score:
#             all_scores[w][year][k] = np.mean(np.array(score[k]).astype(np.float))
#
# print('==========================================================================')
#
# # for w in all_scores:
# #     print('w: ' + w)
# #     for year in all_scores[w]:
# #         print('year: ' + year)
# #         for em in all_scores[w][year]:
# #             print('{}: {}'.format(em, all_scores[w][year][em]))
#
#
# for w in all_scores:
#     for em in emotions:
#         years = [y for y in all_scores[w]]
#         em_over_years = [all_scores[w][year][em] for year in all_scores[w]]
#         plt.plot(years, em_over_years, label=em)
#
#     # plt.legend()
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#                ncol=6, fancybox=True, shadow=True)
#     fig = plt.gcf()
#     fig.set_size_inches(8, 4)
#     # fig.tight_layout()
#     # plt.savefig('{}.png'.format(w))
#     plt.show()
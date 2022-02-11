import matplotlib.pyplot as plt
import pickle
import alyahmor.genelex
import time
import os
import numpy as np


def read_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.readlines()
    words = [w[:-1] for w in words if '\n' in w]
    return words


# # load the neighbors per word of interest, per year
sentiment_words = read_keywords('from_DrFatima/sentiment_keywords.txt')

# mapping each word to an index
word2idx = dict(zip(sentiment_words, list(range(len(sentiment_words)))))

# load the dictionary of emotions
with open('ArSEL_ArSenL_database/emotions_sentiment_ar.pkl', 'rb') as f:
    emotions_sentiment_ar = pickle.load(f)

idx2file = {}
for file in os.listdir('output_diachronic/summaries/sentiment_keywords/'):
    word_idx = int(file.split('_')[0])
    word_year = int(file.split('_')[1][:-4])
    if word_idx not in idx2file:
        idx2file[word_idx] = {}
    idx2file[word_idx][word_year] = file

generator = alyahmor.genelex.genelex()

t1 = time.time()
legal_words = {}
for w in sentiment_words:
    w_idx = word2idx[w]

    for year in idx2file[w_idx]:
        path = os.path.join('output_diachronic/summaries/sentiment_keywords/', idx2file[w_idx][year])
        with open(path, 'rb') as f:
            summary = pickle.load(f)

        words = []
        for n in summary:
            if summary[n] != []:
                corr = summary[n][0]
                if corr in emotions_sentiment_ar:
                    print('{}: {}'.format(corr, emotions_sentiment_ar[corr]['sense'] + ' ==> ' + emotions_sentiment_ar[corr]['sense_definition']))
                    words.append(corr)
                else:
                    if ' ' in corr:
                        corr_l = corr.split(' ')
                        for c in corr_l:
                            found = False
                            try:
                                nouns = generator.generate_forms(c, word_type="noun", vocalized=False)
                                # verbs = generator.generate_forms(c, word_type="verb")
                                for n in nouns:
                                    if n in emotions_sentiment_ar:
                                        print('{}->{}: {}'.format(corr,n, emotions_sentiment_ar[n]['sense'] + ' ==> ' +
                                                              emotions_sentiment_ar[n]['sense_definition']))
                                        words.append(n)
                                        found = True
                                        break
                            except:
                                pass
                            # if not found:
                            #     for v in verbs:
                            #         if v in emotions_sentiment_ar:
                            #             print('{}->{}: {}'.format(corr,v, emotions_sentiment_ar[v]['sense'] + ' ==> ' +
                            #                                   emotions_sentiment_ar[v]['sense_definition']))
                            #             found = True
                            #             break

                            # if not found:
                            #     print('Could not find word {} from {} found in {} summary file of word {}'.format(c, corr, path, w))
                    else:
                        found = False
                        try:
                            nouns = generator.generate_forms(corr, word_type="noun", vocalized=False)
                            # verbs = generator.generate_forms(corr, word_type="verb")
                            for n in nouns:
                                if n in emotions_sentiment_ar:
                                    print('{}->{}: {}'.format(corr, n, emotions_sentiment_ar[n]['sense'] + ' ==> ' +
                                                          emotions_sentiment_ar[n]['sense_definition']))
                                    words.append(n)
                                    found = True
                                    break
                        except:
                            pass
                        # if not found:
                        #     for v in verbs:
                        #         if v in emotions_sentiment_ar:
                        #             print('{}->{}: {}'.format(corr,v, emotions_sentiment_ar[v]['sense'] + ' ==> ' +
                        #                                   emotions_sentiment_ar[v]['sense_definition']))
                        #             found = True
                        #             break

                        # if not found:
                        #     print('Could not find word {} found in {} summary file of word {}'.format(corr, path, w))

        print(words)
        print(len(words), len(summary))

        if w not in legal_words:
            legal_words[w] = {}
        legal_words[w][year] = words

        with open('valid_summaries.pkl', 'wb') as handle:
            pickle.dump(legal_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

t2 = time.time()
with open('valid_summaries.pkl', 'wb') as handle:
    pickle.dump(legal_words, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('time taken: {} mins'.format((t2-t1)/60))

with open('valid_summaries.pkl', 'rb') as handle:
    valid_summaries = pickle.load(handle)

emotions = ['AFRAID', 'AMUSED', 'ANGRY', 'ANNOYED', 'DONT_CARE', 'HAPPY', 'INSPIRED', 'SAD']
scores = {}
for w in sentiment_words:
    w_idx = word2idx[w]
    for em in emotions:
        for year in idx2file[w_idx]:
            if w not in scores:
                scores[w] = {}
            if year not in scores[w]:
                scores[w][year] = {}
            for s in valid_summaries[w][year]:
                if em not in scores[w][year]:
                    scores[w][year][em] = []
                    # scores[w][em][year] = []
                scores[w][year][em].append(emotions_sentiment_ar[s][em])

with open('emotion_scores.pkl', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

plots_folder = 'output_diachronic/summaries/emotion_plots/'
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

avg_scores_yearly = {}
for w in scores:
    avg_scores_yearly[w] = {}
    years = [y for y in scores[w]]
    for y in years:
        avg_scores_yearly[w][y] = {}
        for em in emotions:
            avg = np.mean(scores[w][y][em]) if em in scores[w][y] else 0
            avg_scores_yearly[w][y][em] = avg


for w in sentiment_words:
    years = [y for y in avg_scores_yearly[w]]
    for em in emotions:
        em_over_years = [avg_scores_yearly[w][year][em] for year in years]
        plt.plot(years, em_over_years, label=em)
    plt.legend()
    fig = plt.gcf()
    plt.savefig(os.path.join(plots_folder, '{}.png'.format(w)))
    plt.close()
# emotions_sentiment_ar['عرفات']

import pickle
import qalsadi.lemmatizer
import numpy as np
import matplotlib.pyplot as plt
import os
from manual_corrections_azarbonyad import azarbonyad

lemmer = qalsadi.lemmatizer.Lemmatizer()


def collect_summaries(stored_summaries, emotions_sentiment_ar, years):
    summaries = {}

    for w in stored_summaries:
        summaries[w] = {}
        for year in stored_summaries[w]:
            if year in years: # if we want that year
                summaries[w][year] = []
                summary = stored_summaries[w][year]
                for s in summary:
                    if s in emotions_sentiment_ar:
                        summaries[w][year].append(s)
                    else:
                        if s in azarbonyad:
                            s = azarbonyad[s] if azarbonyad[s] != '' and azarbonyad[s] != '0' else s

                        if ' ' in s:
                            for sp in s.split(' '):
                                if sp in emotions_sentiment_ar:
                                    summaries[w][year].append(sp)
                                else:
                                    sp = lemmer.lemmatize(sp)
                                    if sp in emotions_sentiment_ar:
                                        summaries[w][year].append(sp)
                        else:
                            if s in emotions_sentiment_ar:
                                summaries[w][year].append(s)
                            else:
                                sp = lemmer.lemmatize(s)
                                if sp in emotions_sentiment_ar:
                                    summaries[w][year].append(sp)

                print('processed year: {}'.format(year))

    return summaries


def get_scores(summaries, emotions_aggregated):
    scores = {}
    for w in summaries:
        for em in emotions_aggregated:
            for year in summaries[w]:
                if w not in scores:
                    scores[w] = {}
                if year not in scores[w]:
                    scores[w][year] = {}

                for s in summaries[w][year]:
                    for sub_em in emotions_aggregated[em]:
                        if em not in scores[w][year]:
                            scores[w][year][em] = []
                        # scores[w][em][year] = []
                        scores[w][year][em].append(emotions_sentiment_ar[s][sub_em])
    return scores


# load the dictionary of emotions
with open('../semantic_shifts/words_are_malleable_stability/ArSEL_ArSenL_database/emotions_sentiment_ar.pkl', 'rb') as f:
    emotions_sentiment_ar = pickle.load(f)

with open('year2word2summary_nahar.pickle', 'rb') as handle:
    nahar = pickle.load(handle)

with open('year2word2summary_assafir.pickle', 'rb') as handle:
    assafir = pickle.load(handle)

years_nahar = ['1982', '1983', '1984', '1996', '1997', '1998', '2000', '2005', '2006', '2007', '2008', '2009']
years_assafir = ['1982', '1983', '1984', '1996', '1997', '1998', '2000', '2005', '2006', '2007', '2008', '2009', '2010', '2011']

summaries_nahar = collect_summaries(nahar, emotions_sentiment_ar, years_nahar)
summaries_assafir = collect_summaries(assafir, emotions_sentiment_ar, years_assafir)


# loop over summaries and store the ones found in the emotion lexicon
# with open('azarbonyad_prp.txt', 'r', encoding='utf-8') as f:
#     total_count = 0
#     actual_count = 0
#     everything = f.readlines()
#     summaries = {}
#     for w in everything:
#         if '\t\t\t' in w:
#             if ''.join(w.split(':')[0].split()) != '\'\'':
#                 normal = ''.join([c for c in w if c != '\n' and c != '\t'])
#                 orig = normal.split(':')[1]
#                 corr = normal.split(':')[0]
#                 print('orig: {}, corr: {}'.format(orig, corr))
#                 if ' ' in corr:
#                     spc = corr.split(' ')
#                     for s in spc:
#                         s = s.replace('\'', '')
#                         if s != '-' and s != '':
#                             total_count += 1
#
#                             if s in emotions_sentiment_ar:
#                                 actual_count += 1
#                                 summaries[keyword][year].append(s)
#                             else:
#                                 sl = lemmer.lemmatize(s)
#                                 # sl = stemmer.stem(s)
#                                 if sl in emotions_sentiment_ar:
#                                     print('word: {}, lemma: {}'.format(s, sl))
#                                     actual_count += 1
#                                     summaries[keyword][year].append(sl)
#
#                 else:
#                     s = corr
#                     s = s.replace('\'', '')
#
#                     if s != '-' and s != '':
#                         total_count += 1
#
#                         if s in emotions_sentiment_ar:
#                             actual_count += 1
#                             summaries[keyword][year].append(s)
#                         else:
#                             sl = lemmer.lemmatize(s)
#                             # sl = stemmer.stem(s)
#                             if sl in emotions_sentiment_ar:
#                                 print('word: {}, lemma: {}'.format(s, sl))
#                                 actual_count += 1
#                                 summaries[keyword][year].append(sl)
#
#         elif '\t\t' in w and ':' in w:
#             year = int(''.join(w.split(':')[0].split()).replace('\'', ''))
#             summaries[keyword][year] = []
#
#         elif '\t' in w and ':' in w:
#             keyword = ''.join(w.split(':')[0].split()).replace('\'', '')
#             summaries[keyword] = {}
#
#         else:
#             # do nothing
#             pass
#     # how many words after stemming actually found in the emotion lexicon out of all the words out there
#     print(actual_count/total_count)

# loop over stored summaries and get scores
emotions = ['AFRAID', 'AMUSED', 'ANGRY', 'ANNOYED', 'DONT_CARE', 'HAPPY', 'INSPIRED', 'SAD']
emotions_aggregated = {
    'hopeful': ['HAPPY', 'INSPIRED'],
    'annoyed': ['ANGRY', 'ANNOYED'],
    'afraid': ['AFRAID'],
    'sad': ['SAD']
}
# scores = {}
# for w in summaries:
#     for em in emotions_aggregated:
#         for year in summaries[w]:
#             if w not in scores:
#                 scores[w] = {}
#             if year not in scores[w]:
#                 scores[w][year] = {}
#
#             for s in summaries[w][year]:
#                 for sub_em in emotions_aggregated[em]:
#                     if em not in scores[w][year]:
#                         scores[w][year][em] = []
#                     # scores[w][em][year] = []
#                     scores[w][year][em].append(emotions_sentiment_ar[s][sub_em])
# print(scores)

scores_nahar = get_scores(summaries_nahar, emotions_aggregated)
scores_assafir = get_scores(summaries_assafir, emotions_aggregated)

# loop over scores and get avg values per word per year
avg_scores_yearly_nahar = {}
avg_scores_yearly_assafir = {}

for w in scores_nahar:
    avg_scores_yearly_nahar[w] = {}
    years = [y for y in scores_nahar[w]]
    for y in years:
        avg_scores_yearly_nahar[w][y] = {}
        for em in emotions_aggregated:
            avg = np.mean(scores_nahar[w][y][em]) if em in scores_nahar[w][y] else 0
            avg_scores_yearly_nahar[w][y][em] = avg
print(avg_scores_yearly_nahar)

for w in scores_assafir:
    avg_scores_yearly_assafir[w] = {}
    years = [y for y in scores_assafir[w]]
    for y in years:
        avg_scores_yearly_assafir[w][y] = {}
        for em in emotions_aggregated:
            avg = np.mean(scores_assafir[w][y][em]) if em in scores_assafir[w][y] else 0
            avg_scores_yearly_assafir[w][y][em] = avg
print(avg_scores_yearly_assafir)

# std_scores_yearly = {}
# for w in scores:
#     std_scores_yearly[w] = {}
#     years = [y for y in scores[w]]
#     for y in years:
#         std_scores_yearly[w][y] = {}
#         for em in emotions_aggregated:
#             avg = np.std(scores[w][y][em]) if em in scores[w][y] else 0
#             std_scores_yearly[w][y][em] = avg
# print(std_scores_yearly)
#


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# line plots
plots_folder_lines = 'plots/line_plots/nahar/'
plots_folder_bars = 'plots/bar_plots/'

mkdir(plots_folder_lines)
mkdir(plots_folder_bars)


def slope(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    return m


# line plots
for w in avg_scores_yearly_nahar:
    years = [y for y in avg_scores_yearly_nahar[w]]
    for em in emotions_aggregated:
        em_over_years = [avg_scores_yearly_nahar[w][year][em] for year in years]
        if w == 'arafat':
            points = [(i, v) for i, v in enumerate(em_over_years)]
            slopes = []
            for i in range(len(points)-1):
                slopes.append(slope(points[i][0], points[i][1], points[i+1][0], points[i+1][1]))
            # print('SLOPES: {} : {}'.format(em, slopes))
            print('slopes: {}'.format(em))
            for i, val in enumerate(slopes):
                print('slope {}-{}: {}'.format(years[i], years[i]+1, slopes[i]))
        # std_over_years = [std_scores_yearly[w][year][em] for year in years]
        plt.plot(years, em_over_years,  marker='o', label=em)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=4, fancybox=True, shadow=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.savefig(os.path.join(plots_folder_lines, 'nahar_{}.png'.format(w)))
    plt.close()




# # stacked bar plots
# # width = 0.35
# for w in avg_scores_yearly:
#     fig, ax = plt.subplots()
#     years = [y for y in avg_scores_yearly[w]]
#     i = 0
#     for em in emotions_aggregated:
#         em_over_years = [avg_scores_yearly[w][year][em] for year in years]
#         if i == 0:
#             ax.bar(years, em_over_years, label=em)
#             em_over_years_old = np.array(em_over_years)
#         else:
#             ax.bar(years, em_over_years, bottom=em_over_years_old, label=em)
#             em_over_years_old += np.array(em_over_years)
#         i += 1
#
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#            ncol=4, fancybox=True, shadow=True)
#     # fig = plt.gcf()
#     fig.set_size_inches(10, 5)
#     plt.savefig(os.path.join(plots_folder_bars, '{}.png'.format(w)))
#     plt.close()
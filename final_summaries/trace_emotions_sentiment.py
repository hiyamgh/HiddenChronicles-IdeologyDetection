import alyahmor.genelex
import naftawayh.wordtag
import pickle
import qalsadi.lemmatizer
import numpy as np
import matplotlib.pyplot as plt
import os
# import farasa
# from farasa.stemmer import FarasaStemmer

# load the dictionary of emotions
with open('../semantic_shifts/words_are_malleable_stability/ArSEL_ArSenL_database/emotions_sentiment_ar.pkl', 'rb') as f:
    emotions_sentiment_ar = pickle.load(f)

# define lemmatizer thats actually a stemmer
lemmer = qalsadi.lemmatizer.Lemmatizer()
# stemmer = FarasaStemmer()
# tagger = naftawayh.wordtag.WordTagger()

# loop over summaries and store the ones found in the emotion lexicon
with open('azarbonyad_prp.txt', 'r', encoding='utf-8') as f:
    total_count = 0
    actual_count = 0
    everything = f.readlines()
    summaries = {}
    for w in everything:
        if '\t\t\t' in w:
            if ''.join(w.split(':')[0].split()) != '\'\'':
                normal = ''.join([c for c in w if c != '\n' and c != '\t'])
                orig = normal.split(':')[1]
                corr = normal.split(':')[0]
                print('orig: {}, corr: {}'.format(orig, corr))
                if ' ' in corr:
                    spc = corr.split(' ')
                    for s in spc:
                        s = s.replace('\'', '')
                        if s != '-' and s != '':
                            total_count += 1

                            if s in emotions_sentiment_ar:
                                actual_count += 1
                                summaries[keyword][year].append(s)
                            else:
                                sl = lemmer.lemmatize(s)
                                # sl = stemmer.stem(s)
                                if sl in emotions_sentiment_ar:
                                    print('word: {}, lemma: {}'.format(s, sl))
                                    actual_count += 1
                                    summaries[keyword][year].append(sl)

                else:
                    s = corr
                    s = s.replace('\'', '')

                    if s != '-' and s != '':
                        total_count += 1

                        if s in emotions_sentiment_ar:
                            actual_count += 1
                            summaries[keyword][year].append(s)
                        else:
                            sl = lemmer.lemmatize(s)
                            # sl = stemmer.stem(s)
                            if sl in emotions_sentiment_ar:
                                print('word: {}, lemma: {}'.format(s, sl))
                                actual_count += 1
                                summaries[keyword][year].append(sl)

        elif '\t\t' in w and ':' in w:
            year = int(''.join(w.split(':')[0].split()).replace('\'', ''))
            summaries[keyword][year] = []

        elif '\t' in w and ':' in w:
            keyword = ''.join(w.split(':')[0].split()).replace('\'', '')
            summaries[keyword] = {}

        else:
            # do nothing
            pass
    # how many words after stemming actually found in the emotion lexicon out of all the words out there
    print(actual_count/total_count)

# loop over stored summaries and get scores
emotions = ['AFRAID', 'AMUSED', 'ANGRY', 'ANNOYED', 'DONT_CARE', 'HAPPY', 'INSPIRED', 'SAD']
emotions_aggregated = {
    'hopeful': ['HAPPY', 'INSPIRED'],
    'annoyed': ['ANGRY', 'ANNOYED'],
    'afraid': ['AFRAID'],
    'sad': ['SAD']
}
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
print(scores)

# loop over scores and get avg values per word per year
avg_scores_yearly = {}
for w in scores:
    avg_scores_yearly[w] = {}
    years = [y for y in scores[w]]
    for y in years:
        avg_scores_yearly[w][y] = {}
        for em in emotions_aggregated:
            avg = np.mean(scores[w][y][em]) if em in scores[w][y] else 0
            avg_scores_yearly[w][y][em] = avg
print(avg_scores_yearly)

std_scores_yearly = {}
for w in scores:
    std_scores_yearly[w] = {}
    years = [y for y in scores[w]]
    for y in years:
        std_scores_yearly[w][y] = {}
        for em in emotions_aggregated:
            avg = np.std(scores[w][y][em]) if em in scores[w][y] else 0
            std_scores_yearly[w][y][em] = avg
print(std_scores_yearly)


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# line plots
plots_folder_lines = 'plots/line_plots/'
plots_folder_bars = 'plots/bar_plots/'

mkdir(plots_folder_lines)
mkdir(plots_folder_bars)

# line plots
for w in avg_scores_yearly:
    years = [y for y in avg_scores_yearly[w]]
    for em in emotions_aggregated:
        em_over_years = [avg_scores_yearly[w][year][em] for year in years]
        # std_over_years = [std_scores_yearly[w][year][em] for year in years]
        plt.plot(years, em_over_years,  marker='o', label=em)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=4, fancybox=True, shadow=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.savefig(os.path.join(plots_folder_lines, '{}.png'.format(w)))
    plt.close()

# stacked bar plots

# width = 0.35
for w in avg_scores_yearly:
    fig, ax = plt.subplots()
    years = [y for y in avg_scores_yearly[w]]
    i = 0
    for em in emotions_aggregated:
        em_over_years = [avg_scores_yearly[w][year][em] for year in years]
        if i == 0:
            ax.bar(years, em_over_years, label=em)
            em_over_years_old = np.array(em_over_years)
        else:
            ax.bar(years, em_over_years, bottom=em_over_years_old, label=em)
            em_over_years_old += np.array(em_over_years)
        i += 1

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=4, fancybox=True, shadow=True)
    # fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.savefig(os.path.join(plots_folder_bars, '{}.png'.format(w)))
    plt.close()

# generator = alyahmor.genelex.genelex()
# word = "كتاب"
# # noun_forms = generator.generate_forms(word, word_type="noun", vocalized=False)
# # for form in noun_forms:
# #     print(form)
# verb_forms = generator.generate_forms("كتب", word_type="verb", vocalized=False)
# for v in verb_forms:
#     print(v)

# noun_affix =generator.generate_affix_list(word_type="noun", vocalized=False)
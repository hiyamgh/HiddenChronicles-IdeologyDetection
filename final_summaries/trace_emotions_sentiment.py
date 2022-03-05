import pickle
import qalsadi.lemmatizer
import numpy as np
import matplotlib.pyplot as plt
import os
from manual_corrections_azarbonyad import azarbonyad
import numpy.ma as ma
import matplotlib.cm as cm


def collect_summaries(stored_summaries, emotions_sentiment_ar, years):
    summaries = {}
    for w in stored_summaries:
        summaries[w] = {}
        for year in stored_summaries[w]:
            if year in years:
                summaries[w][year] = []
                summary = stored_summaries[w][year]
                for s in summary:
                    if s in azarbonyad:
                        s = azarbonyad[s] if azarbonyad[s] != '' and azarbonyad[s] != '0' else s

                    else:
                        continue

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
                        scores[w][year][em].append(emotions_sentiment_ar[s][sub_em])
    return scores


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def slope(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    return m


if __name__ == '__main__':
    # define the stemmer
    lemmer = qalsadi.lemmatizer.Lemmatizer()

    # load the dictionary of emotions
    with open('../semantic_shifts/words_are_malleable_stability/ArSEL_ArSenL_database/emotions_sentiment_ar.pkl', 'rb') as f:
        emotions_sentiment_ar = pickle.load(f)

    # load the summaries
    with open('year2word2summary_nahar.pickle', 'rb') as handle:
        nahar = pickle.load(handle)

    with open('year2word2summary_assafir.pickle', 'rb') as handle:
        assafir = pickle.load(handle)

    # years of interest for Nahar archive vs. Assafir archive
    years_nahar = ['1982', '1983', '1984', '1985', '1986', '1987', '1998', '1999', '2000', '2001', '2006', '2007', '2008', '2009']
    years_assafir = ['1982', '1983', '1984', '1985', '1986', '1987', '1998', '1999', '2000', '2001', '2006', '2007', '2008', '2009', '2010', '2011']

    # change summaries to their corrected form (spelling error correction done manually)
    summaries_nahar = collect_summaries(nahar, emotions_sentiment_ar, years_nahar)
    summaries_assafir = collect_summaries(assafir, emotions_sentiment_ar, years_assafir)

    # aggregate emotions
    emotions = ['AFRAID', 'AMUSED', 'ANGRY', 'ANNOYED', 'DONT_CARE', 'HAPPY', 'INSPIRED', 'SAD']
    emotions_aggregated = {
        'hopeful': ['HAPPY', 'INSPIRED'],
        'annoyed': ['ANGRY', 'ANNOYED', 'SAD'],
        'afraid': ['AFRAID'],
    }

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

    # line plots
    plots_folder = 'final_plots/'

    # map arabic words to english words for plotting purposes
    mappings = {
        'اسرائيل': 'israel',
        'السعوديه': 'saudiarabia',
        'المقاومه': 'mukawama',
        'ايران': 'iran',
        'حزب الله': 'hezbollah',
        'رفيق الحريري': 'hariri',
        'سوري': 'syrian',
        'فلسطيني': 'palestinian',
        'منظمه التحرير الفلسطينيه': 'munazama',
        'ياسر عرفات': 'arafat',
    }

    # line plots
    for w in avg_scores_yearly_nahar:
        years = list(range(1982, 2012))
        years = [str(y) for y in years]
        colors = cm.rainbow(np.linspace(0, 1, len(emotions_aggregated)))
        count = 0
        for em in emotions_aggregated:
            em_over_years = [avg_scores_yearly_nahar[w][year][em] if year in avg_scores_yearly_nahar[w] else None for year in years]
            idxs_nonvalid = [i for i, _ in enumerate(em_over_years) if em_over_years[i] == None]
            em_over_years = np.array(em_over_years)
            mc = ma.array(em_over_years)
            mc[idxs_nonvalid] = ma.masked
            plt.plot(years, mc,  marker='o', color=colors[count], label=em + ': nahar')

            em_over_years = [avg_scores_yearly_assafir[w][year][em] if year in avg_scores_yearly_assafir[w] else None for year in years]
            idxs_nonvalid = [i for i, _ in enumerate(em_over_years) if em_over_years[i] == None]
            em_over_years = np.array(em_over_years)
            mc = ma.array(em_over_years)
            mc[idxs_nonvalid] = ma.masked
            plt.plot(years, mc, marker='+', color=colors[count], label=em + ': assafir', linestyle='--')

            count += 1

        plt.ylim([0, 0.5])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        plt.xticks(rotation=45)
        fig = plt.gcf()
        fig.set_size_inches(14, 5)
        plt.savefig(os.path.join(plots_folder, '{}.png'.format(mappings[w.strip()])))
        plt.close()
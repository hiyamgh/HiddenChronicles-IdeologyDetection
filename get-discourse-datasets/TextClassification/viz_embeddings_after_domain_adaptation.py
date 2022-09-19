import pandas as pd

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import plotly
import plotly.graph_objs as go
import numpy as np
from scipy.stats.stats import pearsonr
import argparse
import os


def get_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['shift_index']
    return shifts_dict


def get_embeddings_ocr(vocab_vectors, w, year):
    embeddings = []
    for k in vocab_vectors:
        if w in k and str(year) in k:
            embeddings.append(vocab_vectors[k])
    return embeddings


def get_relative(words, path, years, get_ocr=False):

    with open(path, 'rb') as f:
        vocab_vectors = pickle.load(f, encoding='utf-8')
    wb = 'اسرائيل'

    cosines = {}
    for w in words:
        for year in years:
            if year not in cosines:
                cosines[year] = {}
            year_word = w + '_' + str(year)
            year_word_b = wb + '_' + str(year)
            if get_ocr:
                embeddings = get_embeddings_ocr(vocab_vectors, w, year)
                cs = cosine_similarity(vocab_vectors[year_word_b], np.mean(embeddings, axis=0))[0][0]
            else:
                cs = cosine_similarity(vocab_vectors[year_word_b], vocab_vectors[year_word])[0][0]
            cosines[year]['{}-{}'.format(wb, w)] = cs

    for w in words:
        mcs = []
        w1_w2_1 = '{}-{}'.format(wb, w)
        css = [cosines[1982][w1_w2_1]]
        for year in years[1:]:
            # w1_w2_2 = '{}-{}'.format(wb + '_' + str(year), w + '_' + str(year))
            # w1_w2_2 = '{}-{}'.format(wb + '_' + str(year), w + '_' + str(year))
            mc = abs(cosines[1982][w1_w2_1] - cosines[year][w1_w2_1])
            mcs.append(mc)

            cs = cosines[year][w1_w2_1]
            css.append(cs)

        print('meaning change for {}-{}: {}'.format(wb, w, mcs))
        print('meaning change (cs) for {}-{}: {}'.format(wb, w, css))


def get_cos_dist(words, shifts_dict, path, years):

    cds = []
    cds_gen = []
    meaning_change = []
    shifts = []
    word_list = []
    general = {}
    specific = {}
    with open(path, 'rb') as f:
        vocab_vectors = pickle.load(f, encoding='utf-8')

    for w in words:

        words_emb = {}

        for year in years:
            words_emb[year] = {}
            words_emb[year][w] = []
            year_word = w + '_' + str(year)
            # if year_word in vocab_vectors:
            for k in vocab_vectors:
                if w in k and str(year) in k:
                    words_emb[year][w].append(vocab_vectors[k])
                if w not in general:
                    general[w] = []
                general[w].append(vocab_vectors[year_word])

            print(w, year, np.array(words_emb[year][w]).shape)

            if year == 1982:
                specific[w] = vocab_vectors[year_word]

        for i in range(1, len(years)):
            cs = cosine_similarity(np.mean(words_emb[years[i]][w], axis=0), np.mean(words_emb[years[i-1]][w], axis=0))[0][0]
            cds.append(1 - cs)

        for i in range(len(years)):
            cgs = cosine_similarity(general[w], np.mean(words_emb[years[i]][w], axis=0))
            cds_gen.append(cgs)

        word_list.append(w)
        print(cds)
        print(cds_gen)
        cds = []
        cds_gen = []


    return cds, word_list


def visualize(x,y, words):

    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)


    trace0 = go.Scatter(
        x=x,
        y=y,
        name='Words',
        mode='markers+text',

        marker=dict(

            size=12,
            line=dict(
                width=0.5,
            ),
            opacity=0.75,
        ),
        textfont=dict(color="black", size=19),
        text=words,
        textposition='bottom center'
    )

    trace1 = go.Scatter(
        x=x,
        y=poly1d_fn(x),
        mode='lines',
        name='logistic regression',

    )

    layout = dict(title='Correlation between gs semantic shifts and calculated shifts',
                  yaxis=dict(zeroline=False, title= 'Semantic shift index', title_font = {"size": 20},),
                  xaxis=dict(zeroline=False, title= 'Cosine distance', title_font = {"size": 20},),
                  hovermode='closest',

                  )

    data = [trace0, trace1]
    fig = go.Figure(data=data, layout=layout)
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plotly.offline.plot(fig, filename='visualizations/liverpool.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', type=str,
                        help='Path to output time embeddings',
                        default='E:/nahar-1982-1986-da.pickle')
                        # default='E:/nahar-1982-1986-da.pickle')

    parser.add_argument('--shifts_path', type=str,
                        help='Path to gold standard semantic shifts path',
                        default='data/liverpool/liverpool_shift.csv')
    args = parser.parse_args()

    years = [1982, 1983, 1984, 1985, 1986]
    # years = [1982]

    # shifts_dict = get_shifts(args.shifts_path)
    # words = list(shifts_dict.keys())
    words = ['فلسطين', 'تحرير', 'مقاومه', 'احتلال', 'ارهاب', 'سوريا', 'العراق', 'معارضه', 'حرب', 'مسيح']
    get_relative(words, args.embeddings_path, years, get_ocr=False)
    print('-------------------------------------------------------------------------')
    get_relative(words, args.embeddings_path, years, get_ocr=True)



    # with open(args.embeddings_path, 'rb') as f:
    #     vocab_vectors = pickle.load(f, encoding='utf-8')
    #
    # all = []
    # for w in words:
    #     for year in years:
    #         possible_keys = []
    #         for k in vocab_vectors:
    #             if year in k:
    #                 possible_keys.append(k)
    #
    #         actual_keys = []
    #         for k in possible_keys:
    #             if w in k:
    #                 actual_keys.append(k.split('_')[0])
    #         print('{} - {}: {}'.format(w, year, len(actual_keys)))
    #         all.append(actual_keys)
    #
    # found_in_all = list(set.intersection(*map(set, all)))
    # with open("found_in_all.pickle", "wb") as fp:  # Pickling
    #     pickle.dump(found_in_all, fp)
    # print('number of words found in all years: {}'.format(len(found_in_all)))

    # cds, words = get_cos_dist(words, {}, args.embeddings_path, years)

    #
    #
    # #don't add text to the graph for these words, makes graph less messy
    # dont_draw_list = ['stubbornness', 'tourists', 'semifinals', 'desert', 'talents', 'scorpion',
    #                   'seeded', 'vomit', 'naked', 'strings', 'alternatives', 'leaks', 'bait', 'erect', 'graduate',
    #                   'travel', 'determine', 'explaining', 'soak', 'mouthpuiece', 'congestion', 'revisionism', 'slave',
    #                   'revisonist', 'emotion', 'behaviour', 'listen', 'sentence', 'voice', 'relieved', 'mouthpiece', 'astonishing',
    #                   'participate', 'implied', 'astonishing', 'revisionist', 'patient', 'preventing', 'accomplish', 'narrative',
    #                   'listened', 'egyptian', 'clenched', 'croatian']
    #
    # filtered_words = []
    #
    # for w in words:
    #     if w in dont_draw_list:
    #         filtered_words.append('')
    #     else:
    #         filtered_words.append(w)
    #
    # words = filtered_words
    #
    # print("Pearson coefficient: ", pearsonr(cds, shifts))
    # visualize(cds, shifts, words)

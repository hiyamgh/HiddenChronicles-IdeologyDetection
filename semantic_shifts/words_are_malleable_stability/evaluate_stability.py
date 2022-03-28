import pickle
import os
from itertools import cycle
from bidi import algorithm as bidialg
import arabic_reshaper
# from words_are_malleable import get_stability_combined_one_word
from words_are_malleable4 import get_stability_combined_one_word
import fasttext
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

################### common parameters for all images ###################
plt.rcParams['figure.dpi'] = 300
################### end of common parameters         ###################

import seaborn as sns
import pandas as pd
import csv
import time
import argparse


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def filter_stability_neighbors(stability_neigh, stability_comb):
    """ filter out the unwanted neighbors of words from the neighbors-based approach i.e.
        keep only the words in the common vocabulary
    """
    stability_neigh_filtered = {}
    for k in stability_comb:
        stability_neigh_filtered[k] = stability_neigh[k]
    return stability_neigh_filtered


def print_items(anydict):
    for k, v in anydict.items():
        print(k, v)


def save_heads_tails_all(stabilities_comb, stabilities_neigh, stabilities_lin, n=25, verbose=True, save_heads_tails=True,
                    save_dir=None,
                    file_name=None):
    """ gets the top n most unstable words, and the top n most stable words """
    # sort the stabilities dictionary by increasing order of stabilities (items at the beginning
    # have low stability - items at the end have high stability)
    stabilities_comb = {k: v for k, v in sorted(stabilities_comb.items(), key=lambda item: item[1])}
    stabilities_neigh = {k: v for k, v in sorted(stabilities_neigh.items(), key=lambda item: item[1])}
    stabilities_lin = {k: v for k, v in sorted(stabilities_lin.items(), key=lambda item: item[1])}

    iterations = [(stabilities_comb, 'Combination'), (stabilities_neigh, 'Neighbor-based'),
                  (stabilities_lin, 'Linear-Mapping')
                  ]

    heads_all = {}
    tails_all = {}

    for item in iterations:
        stabilities, name = item[0], item[1]
        # first n items = heads = the most unstable words
        heads = {k: stabilities[k] for k in list(stabilities)[:n]}
        # last n items = tails = the most stable words
        tails = {k: stabilities[k] for k in list(stabilities)[-n:]}

        heads_mod = [(k, v) for k, v in heads.items()]
        tails_mod = [(k, v) for k, v in tails.items()]

        if verbose:
            print('{} - heads:'.format(name))
            print_items(heads)
            print('{} - tails:'.format(name))
            print_items(tails)

        heads_all[name] = heads
        tails_all[name] = tails

        if save_heads_tails:
            mkdir(save_dir)
            with open(os.path.join(save_dir, '{}_{}.csv'.format(file_name, name)), 'w', encoding='utf-8-sig', newline='') as f:
                r = csv.writer(f)
                r.writerow(['heads', 'tails'])

                for i in range(len(heads_mod)):
                    r.writerow([heads_mod[i][0] + "," + str(heads_mod[i][1]), tails_mod[i][0] + "," + str(tails_mod[i][1])])

    return heads_all, tails_all


def get_heads_tails(stabilities, n, verbose=True):
    """ gets the top n most unstable words, and the top n most stable words """
    # sort the stabilities dictionary by increasing order of stabilities (items at the beginning
    # have low stability - items at the end have high stability)
    stabilities = {k: v for k, v in sorted(stabilities.items(), key=lambda item: item[1])}

    # first n items = heads = the most unstable words
    heads = {k: stabilities[k] for k in list(stabilities)[:n]}
    # last n items = tails = the most stable words
    tails = {k: stabilities[k] for k in list(stabilities)[-n:]}
    if verbose:
        print('heads:')
        print_items(heads)
        print('tails:')
        print_items(tails)
    return heads, tails


def get_stability_words(stabilities, words):
    """ prints the stability value of each word """
    for w in words:
        if w in stabilities:
            print('{}: {}'.format(w, str(stabilities[w])))
        else:
            print('word {} not found in teh dictionary'.format(w))


def jaccard_similarity(listoflists):
    inter = set.intersection(*map(set, listoflists))
    un = set().union(*listoflists)
    return float(len(inter) / len(un))

# ax[0].set_xlabel("Combined" if i == 0 else "Neighbor-based" if i == 1 else "Linear-Mapping")


def generate_stability_heatmap(words, stability_dicts_combined, stability_dicts_neighbor,
                               stability_dicts_linear,
                               years, save_dir, fig_name):
    yticks = [bidialg.get_display(arabic_reshaper.reshape(w)) for w in words]
    numxticks = len(stability_dicts_combined)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    data_comb, data_neigh, data_lin = [], [], []
    for w in words:
        stab_vals_comb, stab_vals_neigh, stab_vals_lin = [], [], []
        for j in range(len(years)):
            stab_vals_comb.append(stability_dicts_combined[j][w])
            stab_vals_neigh.append(stability_dicts_neighbor[j][w])
            stab_vals_lin.append(stability_dicts_linear[j][w])
        data_comb.append(stab_vals_comb)
        data_neigh.append(stab_vals_neigh)
        data_lin.append(stab_vals_lin)

    data_comb = np.array(data_comb)
    sns.heatmap(data_comb, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=False, ax=ax[0])
    ax[0].set_xlabel("Combined")
    ax[0].set_xticks(list(range(numxticks)))
    ax[0].set_xticklabels(years, rotation=90)

    data_neigh = np.array(data_neigh)
    sns.heatmap(data_neigh, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=False, ax=ax[1])
    ax[1].set_xlabel("Neighbor-based")
    ax[1].set_xticks(list(range(numxticks)))
    ax[1].set_xticklabels(years, rotation=90)

    data_lin = np.array(data_lin)
    sns.heatmap(data_lin, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=False, ax=ax[2])
    ax[2].set_xticks(list(range(numxticks)))
    ax[2].set_xlabel("Linear-Mapping")
    ax[2].set_xticklabels(years, rotation=90)

    mkdir(save_dir)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, fig_name + '_stabilities_heatmap.png'))
    plt.close()


def plot_jaccard_similarity_tails(stability_dicts_combined, stability_dicts_neighbor, stability_dicts_linear, n_sizes,
                                  save_dir, fig_name):
    """ get the jaccard similarity between the tails of the different stability
        approaches for different sizes of n. Ideally, because words in tails should be stable,
        they must be present in the tails of any corpus used.
    """
    jaccard_sims_comb, jaccard_sims_neigh, jaccard_sims_lin = [], [], []
    for n in n_sizes:
        # get the jaccard similarity over the "combined"-based dictionaries
        all_tails = []
        for stab_dict in stability_dicts_combined:
            _, tails = get_heads_tails(stab_dict, n, verbose=False)
            all_tails.append(tails)
        jaccard = jaccard_similarity(all_tails)
        jaccard_sims_comb.append(jaccard)

        # get the jaccard similarity over the "neighbor"-based dictionaries
        all_tails = []
        for stab_dict in stability_dicts_neighbor:
            _, tails = get_heads_tails(stab_dict, n, verbose=False)
            all_tails.append(tails)
        jaccard = jaccard_similarity(all_tails)
        jaccard_sims_neigh.append(jaccard)

        # get the jaccard similarity over the "linear"-based dictionaries
        all_tails = []
        for stab_dict in stability_dicts_linear:
            _, tails = get_heads_tails(stab_dict, n, verbose=False)
            all_tails.append(tails)
        jaccard = jaccard_similarity(all_tails)
        jaccard_sims_lin.append(jaccard)

    lines = ["--", "-.", ":"]
    linecycler = cycle(lines)
    plt.figure()
    for i in range(len(lines)):
        if i == 0:
            plt.plot(list(n_sizes), jaccard_sims_comb, next(linecycler), label="Combination")
        elif i == 1:
            plt.plot(list(n_sizes), jaccard_sims_neigh, next(linecycler), label="Neighbors-based")
        else:
            plt.plot(list(n_sizes), jaccard_sims_lin, next(linecycler), label="Linear-Mapping")
    plt.legend()
    plt.xlabel('tail sizes')
    plt.ylabel('jaccard similarity')
    plt.xlim([n_sizes[0], n_sizes[-1]])
    # plt.ylim([0, max(max(jaccard_sims_comb), max(jaccard_sims_neigh), max(jaccard_sims_lin))])
    plt.ylim([0, 0.1])
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '_jaccard_similarities.png'))
    plt.close()


def plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, words,
                           save_dir, fig_name):
    deltas_neigh, deltas_lin = [], []
    words_decoded = []
    for w in words:
        drneigh = ranks_neigh[w] - ranks_comb[w]  # get the delta rank
        drlin = ranks_lin[w] - ranks_comb[w]  # get the delta rank
        deltas_neigh.append(drneigh)  # add to list
        deltas_lin.append(drlin)  # add to list
        words_decoded.append(
            bidialg.get_display(arabic_reshaper.reshape(w)))  # decode arabic word to make it appear in matplotlib
    plt.bar(words_decoded, deltas_neigh, label='Combination vs. Neighbor-based')
    plt.bar(words_decoded, deltas_lin, label='Combination vs. Linear Mapping', bottom=deltas_neigh)
    plt.xticks(rotation=90)
    plt.ylabel(r'$\Delta$' + 'rank')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    fig.tight_layout()
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '_delta_ranks.png'))
    plt.close()


def get_ranks(stability_combined, stability_neighbors, stability_linear):
    # sort the stabilities dictionary by increasing order of stabilities (items at the beginning
    # have low stability - items at the end have high stability)
    stability_combined = {k: v for k, v in sorted(stability_combined.items(), key=lambda item: item[1])}
    stability_neighbors = {k: v for k, v in sorted(stability_neighbors.items(), key=lambda item: item[1])}
    stability_linear = {k: v for k, v in sorted(stability_linear.items(), key=lambda item: item[1])}

    values_combined = list(stability_combined.values())
    values_neighbor = list(stability_neighbors.values())
    values_linear = list(stability_linear.values())

    ranks_combined, ranks_neigh, ranks_lin = [], [], []

    ranks_combined.append(1)  # for the first value, its rank is 1
    ranks_neigh.append(1)  # for the first value, its rank is 1
    ranks_lin.append(1)

    # get the rankings per value for the combined
    rank = 1
    for i in range(1, len(values_combined[1:]) + 1):
        if round(values_combined[i], 5) == round(values_combined[i - 1], 5):
            ranks_combined.append(rank)
        else:
            rank += 1
            ranks_combined.append(rank)
    print(len(ranks_combined) == len(values_combined))

    # get the rankings per value for the neighbors
    rank = 1
    for i in range(1, len(values_neighbor[1:]) + 1):
        if round(values_neighbor[i], 8) == round(values_neighbor[i - 1], 8):
            ranks_neigh.append(rank)
        else:
            rank += 1
            ranks_neigh.append(rank)
    print(len(ranks_neigh) == len(values_neighbor))

    # get the rankings per value for the linear
    rank = 1
    for i in range(1, len(values_linear[1:]) + 1):
        if round(values_linear[i], 8) == round(values_linear[i - 1], 8):
            ranks_lin.append(rank)
        else:
            rank += 1
            ranks_lin.append(rank)
    print(len(ranks_lin) == len(values_linear))

    ranks_combined = dict(zip(list(stability_combined.keys()), ranks_combined))
    ranks_neigh = dict(zip(list(stability_neighbors.keys()), ranks_neigh))
    ranks_lin = dict(zip(list(stability_linear.keys()), ranks_lin))

    return ranks_combined, ranks_neigh, ranks_lin


def get_contrastive_viewpoint_summary(w, n, k, models, mat_name, dir_name_matrices,
                                      save_dir, file_name, viewpoints_names,
                                      thresh=0.5):
    """ get a contrastive viewpoint summary of a word of length n. For a certain
        word:
        1. we get its top k nearest neighbors.
        2. Then for each nearest neighbor, we add it into the summary if its stability is equal to or less than a certain threshold.
    """
    summary1, summary2, summary3 = [], [], []
    all_summaries = []

    nns1 = [n[1] for n in models[0].get_nearest_neighbors(w, k)]
    nns2 = [n[1] for n in models[1].get_nearest_neighbors(w, k)]

    if len(models) > 2:
        nns3 = [n[1] for n in models[2].get_nearest_neighbors(w, k)]
        count = 0
        for nn in nns3:
            if count == n:
                break
            st = get_stability_combined_one_word(w=nn, models=models, models_names=viewpoints_names, mat_name=mat_name, dir_name_matrices=dir_name_matrices, k=k)

            if abs(st) <= thresh:
                summary3.append((st, nn))
                count += 1

    count = 0
    for nn in nns1:
        if count == n:
            break

        st = get_stability_combined_one_word(w=nn, models=models, models_names=viewpoints_names, mat_name=mat_name, dir_name_matrices=dir_name_matrices, k=k)
        if st <= thresh:
            summary1.append((st, nn))
            count += 1
    all_summaries.append(summary1)

    count = 0
    for nn in nns2:
        if count == n:
            break

        st = get_stability_combined_one_word(w=nn, models=models, models_names=viewpoints_names, mat_name=mat_name, dir_name_matrices=dir_name_matrices, k=k)

        if abs(st) <= thresh:
            summary2.append((st, nn))
            count += 1
    all_summaries.append(summary2)
    if len(models) > 2:
        all_summaries.append(summary3)

    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.txt'.format(file_name)), 'a', encoding='utf-8') as f:
        for j in range(len(viewpoints_names)):
            f.write('w: {}\n'.format(w))
            f.write('viewpoint {}\n'.format(viewpoints_names[j]))
            for i, s in enumerate(all_summaries[j]):
                if i % 10 != 0 or i == 0:
                    f.write(s[1] + ", ")
                else:
                    f.write(s[1] + "\n")
            f.writelines('\n')
            f.writelines('\n')
            f.writelines('-------------------------------------------------------------------------------------------\n')
        f.writelines('==================================================== END OF SUMMARIES FOR WORD {} =================================================================================='.format(w))
        f.writelines('\n')
    f.close()

    # if summary3 is [], it will be returned as such
    return summary1, summary2, summary3


def perform_paired_t_test(ranks_comb, ranks_neigh, ranks_lin, save_dir, file_name):
    # get test-statistic and p-value results
    result_comb_neigh = stats.ttest_rel(list(ranks_comb.values()), list(ranks_neigh.values()))
    result_comb_lin = stats.ttest_rel(list(ranks_comb.values()), list(ranks_lin.values()))

    # get the average rank of each method
    avg_rank_comb = np.mean(list(ranks_comb.values()))
    avg_rank_neigh = np.mean(list(ranks_neigh.values()))
    avg_rank_lin = np.mean(list(ranks_lin.values()))

    print('avg rank combined: {}'.format(np.mean(list(ranks_comb.values()))))
    print('avg rank neighbors: {}'.format(np.mean(list(ranks_neigh.values()))))
    print('avg rank linear: {}'.format(np.mean(list(ranks_lin.values()))))

    df = pd.DataFrame(columns=['Avg. Rank Comb', 'Avg. Rank Neigh', 'Avg. Rank Lin',
                               'ttest_comb_neigh', 'ttest_comb_lin',
                               'pval_comb_neigh', 'pval_comb_lin'])

    df = df.append({
        'Avg. Rank Comb': avg_rank_comb,
        'Avg. Rank Neigh': avg_rank_neigh,
        'Avg. Rank Lin': avg_rank_lin,
        'ttest_comb_neigh': result_comb_neigh[0],
        'ttest_comb_lin': result_comb_lin[0],
        'pval_comb_neigh': result_comb_neigh[1],
        'pval_comb_lin': result_comb_lin[1]
    }, ignore_index=True)

    if result_comb_neigh[1] < 0.05:
        print(result_comb_neigh)
        print('accept H1 for comb-neigh')
    else:
        print(result_comb_neigh)
        print('accept H0 for comb-neigh')

    if result_comb_lin[1] < 0.05:
        print(result_comb_lin)
        print('accept H1 for comb-lin')
    else:
        print(result_comb_lin)
        print('accept H0 for comb-lin')

    mkdir(save_dir)
    df.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)


def read_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.readlines()
    words = [w[:-1] for w in words if '\n' in w]
    words = [w for w in words if w.strip() != '']
    return words


def save_summary(original_word, summary, year, save_dir, filename):
    mkdir(save_dir)
    with open(os.path.join(save_dir, filename + '.txt'), 'a', encoding='utf-8') as f:
        f.write('\nplease manually correct the summaries below for the word: {} in year: {}\n'.format(original_word, year))
        words = [w[1] for w in summary] # words are the neighbors
        for w in words:
            f.write('{}:\n'.format(w))
        f.write('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    f.close()


# def store_summary_corrections(summary, save_dir, filename):
#     t1 = time.time()
#     corr = Corrector()
#     corrections = {}
#     words = [w[1] for w in summary]
#     for w in words:
#         corrections[w] = []
#         print('w: {}'.format(w))
#
#         # if the word is correct then no need for doing any corrections
#         check = input('does the word {} have a correct spelling ?'.format(w))
#         if check == 'y':
#             corrections[w].append(w)
#             continue
#
#         # if the word contains a space then its a phrase (i.e more than one word)
#         if ' ' in w:
#             corrc = corr.contextual_correct(w)
#             print('correction: {}'.format(corrc))
#             include = input('is {} a correct word to include?'.format(corrc))
#             if include == 'y':
#                 corrections[w].append(corrc)
#         else:
#             for i in range(1, len(w)):
#                 new_str = w[:i] + ' ' + w[i:]
#                 print(new_str)
#                 corrc = corr.contextual_correct(new_str)
#                 include = input('is {} a correct word to include?'.format(corrc))
#                 if include == 'y':
#                     corrections[w].append(corrc)
#                     cont = input('would you like to continue (y) or move to another word (n)?')
#                     if cont == 'y':
#                         continue
#                     else:  # move to another word
#                         break
#                 print('------------------------------')
#             if corrections[w] == []:
#                 addwordmanually = input('Do you want to manually add the correction for word {}'.format(w))
#                 if addwordmanually == 'y':
#                     wordfromuser = input('Please insert the correction: ')
#                     corrections[w].append(wordfromuser)
#                     print('------------------------------')
#         print('=================================================')
#     t2 = time.time()
#     print('time taken: {} mins'.format((t2 - t1) / 60))
#     for k, v in corrections.items():
#         print('{}: {}'.format(k, v))
#         print('???????????????????????????')
#     mkdir(save_dir)
#     with open(os.path.join(save_dir, '{}.pkl'.format(filename)), 'wb') as f:
#         pickle.dump(corrections, f)
#     return corrections


def plot_stabilities_over_time(words, stabilities_over_time, mode, save_dir, fig_name):
    for w in words:
        stabilities = []
        for tp in stabilities_over_time: # tp meaning time_point
            st = stabilities_over_time[tp][w]
            stabilities.append(st)
        w_proc = bidialg.get_display(arabic_reshaper.reshape(w))
        plt.plot(list(stabilities_over_time.keys()), stabilities, label=w_proc)

    type_analysis = 'diachronic' if mode[:2] == 'd-' else 'synchronic'
    archive = mode[2:] if mode != 's' else None
    # word_proc = bidialg.get_display(arabic_reshaper.reshape(w))
    if archive is not None:
        xlab = 'time points of the stability in {} archive - {} analysis'.format(archive, type_analysis)
    else:
        xlab = 'time points of the stability - {} analysis'.format(type_analysis)
    plt.xlabel(xlab)
    plt.ylabel('stability')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tick_params(axis='x', which='major', pad=1)
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '.png'))
    plt.savefig(os.path.join(save_dir, fig_name + '.pdf'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path1', default='D:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/', help='path to trained models files')
    parser.add_argument('--models_path2', default='D:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/', help='path to trained models files of viewpoint 2. If not None, then analysis will be synchronic, else, analysis will be diachronic')
    parser.add_argument('--models_path3', default=None, help='path to trained models files of viewpoint 3. If not None, then analysis will be synchronic, else, analysis will be diachronic')
    parser.add_argument('--keywords_path', default='from_DrFatima/sentiment_keywords.txt')
    parser.add_argument("--mode", default="d-assafir", help="mode: \'d-archivename\' for diachronic, \'s\' for synchronic")

    args = parser.parse_args()

    path1 = args.models_path1
    path2 = args.models_path2
    path3 = args.models_path3

    # for sentiment
    sentiment_words = read_keywords(args.keywords_path)

    # absolute prefix of any path on the hpc cluster
    # prefix = '/scratch/7613491_hkg02/political_discourse_mining_hiyam/semantic_shifts_modified/'

    # key is mode, value is start and end years
    dict_years = {
        'd-nahar': {
            'start': 1983,
            'end': 2009,
            'years': [[y - 1, y] for y in list(range(1983, 2010))],
            'paths': ['stability_diachronic/nahar/nahar_{}_nahar_{}/'.format(y - 1, y) for y in list(range(1983, 2010))],
            'viewpoints': [['nahar_{}'.format(y - 1), 'nahar_{}'.format(y)] for y in list(range(1983, 2010))],
            'models': [['{}.bin'.format(y - 1), '{}.bin'.format(y)] for y in list(range(1983, 2010))],
            'time_points': ['{}-{}'.format(y - 1, y) for y in list(range(1983, 2010))]
        },
        'd-assafir': {
            'start': 1983,
            'end': 2011,
            'years': [[y-1, y] for y in list(range(1983, 2012))],
            'paths': ['stability_diachronic/assafir/assafir_{}_assafir_{}/'.format(y - 1, y) for y in list(range(1983, 2012))],
            'viewpoints': [['assafir_{}'.format(y - 1), 'assafir_{}'.format(y)] for y in list(range(1983, 2012))],
            'models': [['{}.bin'.format(y - 1), '{}.bin'.format(y)] for y in list(range(1983, 2012))],
            'time_points': ['{}-{}'.format(y - 1, y) for y in list(range(1983, 2012))]
        },
        'd-hayat': {
            'start': 1988,
            'end': 2000,
            'years': [[y - 1, y] for y in list(range(1989, 2001))],
            'paths': ['stability_diachronic/hayat/hayat_{}_hayat_{}/'.format(y - 1, y) for y in list(range(1989, 2001))],
            'viewpoints': [['hayat_{}'.format(y - 1), 'hayat_{}'.format(y)] for y in list(range(1989, 2001))],
            'models': [['{}.bin'.format(y - 1), '{}.bin'.format(y)] for y in list(range(1989, 2001))],
            'time_points': ['{}-{}'.format(y - 1, y) for y in list(range(1989, 2001))]
        },
        's': {
            'start': 1988,
            'end': 2000,
            'years': [[y, y, y] for y in list(range(1988, 2001))],
            'paths': ['stability_synchronic/nahar_{}_assafir_{}_hayat_{}/'.format(y, y, y) for y in list(range(1988, 2001))],
            'viewpoints': [['nahar_{}'.format(y), 'assafir_{}'.format(y), 'hayat_{}'.format(y)] for y in list(range(1988, 2001))],
            'models': [['{}.bin'.format(y), '{}.bin'.format(y), '{}.bin'.format(y)] for y in list(range(1988, 2001))],
            'time_points': ['{}'.format(y) for y in list(range(1988, 2001))]
        }
    }

    mode = args.mode
    paths = dict_years[mode]['paths']
    stabilities_over_time = {}
    results_dir = 'evaluate_stability/{}/'.format(mode)

    for i, path in enumerate(paths):
        if os.path.exists(path):
            dict_combined = os.path.join(path + 'k100/', 'stabilities_combined.pkl')
            print('path: {}'.format(path))

            models2load = dict_years[mode]['models'][i]
            viewpoints = dict_years[mode]['viewpoints'][i]
            years2load = dict_years[mode]['years'][i]
            time_point = dict_years[mode]['time_points'][i]

            stabilities_over_time[time_point] = {}

            if os.path.exists(dict_combined):
                with open(dict_combined, 'rb') as handle: # load pickle file of stability dictionary
                    stabilities_comb = pickle.load(handle)
                    print('loaded the stability dictionary for time point {} in mode: {}'.format(time_point, mode))

            stabilities_over_time[time_point] = stabilities_comb # store the stability values for a particular time point

            models = [] # to store loaded models inside an array to pass to the get_summaries method
            model1 = fasttext.load_model(os.path.join(path1, '{}'.format(models2load[0])))
            model2 = fasttext.load_model(os.path.join(path2, '{}'.format(models2load[1])))

            models.append(model1)
            models.append(model2)
            if len(models2load) > 2:
                model3 = fasttext.load_model(os.path.join(path3, '{}'.format(models2load[2])))
                models.append(model3)

            dir_name_matrices = '{}/linear_numsteps80000/matrices/'.format(path)

            years_checked = {}
            for z, w in enumerate(sentiment_words):

                summary_v1, summary_v2, summary_v3 = get_contrastive_viewpoint_summary(w, n=20, k=100,
                                                                           models=models,
                                                                           mat_name='trans',
                                                                           dir_name_matrices=dir_name_matrices,
                                                                           save_dir=results_dir + 'summaries/',
                                                                           file_name='sentiment_keywords',
                                                                           viewpoints_names=viewpoints,
                                                                           thresh=0.5)
                if w not in years_checked:
                    years_checked[w] = []

                if years2load[0] not in years_checked[w]:
                    save_summary(original_word=w, summary=summary_v1, year=years2load[0], save_dir=results_dir + 'summaries/', filename='summaries_azarbonyad')
                    years_checked[w].append(years2load[0])

                if years2load[1] not in years_checked[w]:
                    save_summary(original_word=w, summary=summary_v2, year=years2load[1], save_dir=results_dir + 'summaries/', filename='summaries_azarbonyad')
                    years_checked[w].append(years2load[1])

                if summary_v3 != []:
                    if years2load[2] not in years_checked[w]:
                        save_summary(original_word=w, summary=summary_v3, year=years2load[2], save_dir=results_dir + 'summaries/', filename='summaries_azarbonyad')
                        years_checked[w].append(years2load[2])

    # a dictionary mapping word in Arabic to a 'temporary name' in English
    # to make it easier to save plots and retrieve them
    # later on in latex (for reporting)
    mapar2en = {
        'الولايات المتحده الاميركيه': 'UnitedStatesofAmerica',
        'اميركا': 'America',
        'اسرائيل': 'Israel',
        'فلسطيني': 'Palestinian',
        'حزب الله': 'Hezbollah',
        'المقاومه': 'Resistance',
        'سوري': 'Syrian',
        'منظمه التحرير الفلسطينيه': 'PalestinianLiberationOrganization',
        'ايران': 'Iran',
        'السعوديه': 'Saudiya'
    }

    words_batch1 = ['فلسطيني', 'منظمه التحرير الفلسطينيه', 'المقاومه']
    words_batch2 = ['السعوديه', 'الولايات المتحده الاميركيه', 'اميركا']
    words_batch3 = ['اسرائيل']
    words_batch4 = ['حزب الله', 'المقاومه', 'سوري',  'ايران']

    # plot the stability as a function of time for each word
    # for w in sentiment_words:
    #     plot_stabilities_over_time(w, stabilities_over_time, mode, results_dir + 'stability_plots/', mapar2en[w])

    plot_stabilities_over_time(words_batch1, stabilities_over_time, mode, results_dir + 'stability_plots/', 'palestine_related')
    plot_stabilities_over_time(words_batch2, stabilities_over_time, mode, results_dir + 'stability_plots/', 'america_related')
    plot_stabilities_over_time(words_batch3, stabilities_over_time, mode, results_dir + 'stability_plots/', 'israel_related')
    plot_stabilities_over_time(words_batch4, stabilities_over_time, mode, results_dir + 'stability_plots/', 'syrian_related')



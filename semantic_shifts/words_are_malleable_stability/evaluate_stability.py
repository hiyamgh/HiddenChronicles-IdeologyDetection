import pickle
import os
from itertools import cycle
from bidi import algorithm as bidialg
import arabic_reshaper
from words_are_malleable import get_stability_combined_one_word
import fasttext
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
# import sentiment word analysis related topics
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from ar_corrector.corrector import Corrector
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


# cannot decide yet on summary because our words are oov and nearest
# neighs of oov are also oov, should we also get stability values
# for these ?

def get_contrastive_viewpoint_summary(w, n, k, model1, model2, mat_name, dir_name_matrices,
                                      save_dir, file_name, viewpoint1_name='1982', viewpoint2_name='1983',
                                      thresh=0.6):
    """ get a contrastive viewpoint summary of a word of length n. For a certain
        word, we get its top k nearest neighbors. Then for each nearest neighbor, we add it
        into the summary if its stability is less than a certain threshold.
    """
    summary1, summary2 = [], []

    nns1 = [n[1] for n in model1.get_nearest_neighbors(w, k)]
    nns2 = [n[1] for n in model2.get_nearest_neighbors(w, k)]

    count = 0
    for nn in nns1:
        if count == n:
            break
        st = get_stability_combined_one_word(w=nn, model1=model1, model2=model2, mat_name=mat_name,
                                             dir_name_matrices=dir_name_matrices)

        if st <= thresh:
            summary1.append((st, nn))
            count += 1
    count = 0
    for nn in nns2:
        if count == n:
            break
        st = get_stability_combined_one_word(w=nn, model1=model1, model2=model2, mat_name=mat_name,
                                             dir_name_matrices=dir_name_matrices)

        if st <= thresh:
            summary2.append((st, nn))
            count += 1

    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.txt'.format(file_name)), 'a', encoding='utf-8') as f:
        f.write('w: {}\n'.format(w))
        f.write('viewpoint {}\n'.format(viewpoint1_name))
        for i, s in enumerate(summary1):
            if i % 10 != 0 or i == 0:
                f.write(s[1] + ", ")
            else:
                f.write(s[1] + "\n")
        f.writelines('\n')
        f.writelines('\n')
        f.writelines('-------------------------------------------------------------------------------------------\n')
        f.write('w: {}\n'.format(w))
        f.write('viewpoint {}\n'.format(viewpoint2_name))
        for i, s in enumerate(summary2):
            if i % 10 != 0 or i == 0:
                f.write(s[1] + ", ")
            else:
                f.write(s[1] + "\n")
        f.writelines('\n')
        f.writelines('\n')
        f.writelines('========================================================================================================================================')
        f.writelines('\n')
    f.close()

    return summary1, summary2


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
    return words


def store_summary_corrections(summary, save_dir, filename):
    t1 = time.time()
    corr = Corrector()
    corrections = {}
    words = [w[1] for w in summary]
    for w in words:
        corrections[w] = []
        print('w: {}'.format(w))

        # if the word is correct then no need for doing any corrections
        check = input('does the word {} have a correct spelling ?'.format(w))
        if check == 'y':
            corrections[w].append(w)
            continue

        # if the word contains a space then its a phrase (i.e more than one word)
        if ' ' in w:
            corrc = corr.contextual_correct(w)
            print('correction: {}'.format(corrc))
            include = input('is {} a correct word to include?'.format(corrc))
            if include == 'y':
                corrections[w].append(corrc)
        else:
            for i in range(1, len(w)):
                new_str = w[:i] + ' ' + w[i:]
                print(new_str)
                corrc = corr.contextual_correct(new_str)
                include = input('is {} a correct word to include?'.format(corrc))
                if include == 'y':
                    corrections[w].append(corrc)
                    cont = input('would you like to continue (y) or move to another word (n)?')
                    if cont == 'y':
                        continue
                    else:  # move to another word
                        break
                print('------------------------------')
            if corrections[w] == []:
                addwordmanually = input('Do you want to manually add the correction for word {}'.format(w))
                if addwordmanually == 'y':
                    wordfromuser = input('Please insert the correction: ')
                    corrections[w].append(wordfromuser)
                    print('------------------------------')
        print('=================================================')
    t2 = time.time()
    print('time taken: {} mins'.format((t2 - t1) / 60))
    for k, v in corrections.items():
        print('{}: {}'.format(k, v))
        print('???????????????????????????')
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(filename)), 'wb') as f:
        pickle.dump(corrections, f)
    return corrections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path1',
                        default='D:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/',
                        help='path to trained models files')
    parser.add_argument('--models_path2',
                        default=None,
                        help='path to trained models files of viewpoint 2. If not None, then analysis will be synchronic, else, analysis will be diachronic')
    parser.add_argument('--keywords_paths', default='from_DrFatima_cleaned/ethnicities.txt;from_DrFatima_cleaned/ideologies.txt;from_DrFatima_cleaned/political_parties.txt;'
                                                    'from_DrFatima_cleaned/politicians.txt;from_DrFatima_cleaned/israeli_leaders.txt')
    parser.add_argument('--keywords_names', default='ethnicities;ideologies;political_parties;politicians;israeli_leaders')
    parser.add_argument('--start_year', default=1982)
    parser.add_argument('--end_year', default=2009)

    parser.add_argument("--model1", default='2007.bin', help="model 1 file name")
    parser.add_argument("--model2", default='2007.bin', help="model 2 file name")
    parser.add_argument("--model1_name", default="nahar_07", help="string to name model 1 - used for saving results")
    parser.add_argument("--model2_name", default="assafir_07", help="string to name model 2 - used for saving results")

    path1 = 'D:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/'
    # path2 = 'E:/fasttext_embeddings/assafir/'
    path2 = path1

    ethnicities = read_keywords('from_DrFatima_cleaned/ethnicities.txt')
    ideologies = read_keywords('from_DrFatima_cleaned/ideologies.txt')
    political_parties = read_keywords('from_DrFatima_cleaned/political_parties.txt')
    politicians = read_keywords('from_DrFatima_cleaned/politicians.txt')
    israeli_leaders = read_keywords('from_DrFatima_cleaned/israeli_leaders.txt')

    # for sentiment
    sentiment_words = read_keywords('from_DrFatima/sentiment_keywords.txt')

    years = list(range(1983, 2009))
    yearsforfigs = ['{}-{}'.format(y-1, y) for y in years]
    fig_name_prefixes = [
        'nahar_{}_{}'.format(y - 1, y) for y in years
    ]
    fig_name_general_prefix = 'nahar'

    paths = [
        'E:/fasttext_embeddings/results_diachronic/nahar_{}_nahar_{}/t1k100/'.format(y - 1, y) for y in years
    ]

    stability_dicts_combined = []
    stability_dicts_neighbor = []
    stability_dicts_linear = []
    results_dir = 'output_diachronic/'

    # to store all summaries of all words
    all_summaries = {}
    for i, path in enumerate(paths):
        dict_combined = os.path.join(path, 'stabilities_combined.pkl')
        dict_neighbor = os.path.join(path, 'stabilities_neighbor.pkl')
        dict_linear = os.path.join(path[:-7], 'stabilities_linear.pkl')
        print('path: {}'.format(path))

        if os.path.exists(dict_combined):
            print('combined:')
            # load pickle file of stabilities
            with open(dict_combined, 'rb') as handle:
                stabilities_comb = pickle.load(handle)
                stability_dicts_combined.append(stabilities_comb)
                # get_stability_words(stabilities, words)
                # get_heads_tails(stabilities, n=50)
            print('================================================================')

        if os.path.exists(dict_neighbor):
            print('neighbor:')
            # load pickle file of stabilities
            with open(dict_neighbor, 'rb') as handle:
                stabilities_neigh = pickle.load(handle)
                stabilities_neigh = filter_stability_neighbors(stabilities_neigh, stabilities_comb)
                stability_dicts_neighbor.append(stabilities_neigh)
                # get_stability_words(stabilities, words)
                # get_heads_tails(stabilities, n=50)
            print('================================================================')

        if os.path.exists(dict_linear):
            print('linear:')
            # load pickle file of stabilities
            with open(dict_linear, 'rb') as handle:
                stabilities_lin = pickle.load(handle)
                stabilities_lin = filter_stability_neighbors(stabilities_lin, stabilities_comb)
                stability_dicts_linear.append(stabilities_lin)
            print('================================================================')

        model1 = fasttext.load_model(os.path.join(path1, '{}.bin'.format(years[i-1])))
        model2 = fasttext.load_model(os.path.join(path2, '{}.bin'.format(years[i])))


        dir_name_matrices = 'D:/fasttext_embeddings/results_diachronic/nahar_{}_nahar_{}/linear_numsteps80000/matrices/'.format(str(years[i]-1), str(years[i]))

        # create a mapping between word and a numeric index
        word2idx = dict(zip(sentiment_words, list(range(len(sentiment_words)))))
        for z, w in enumerate(sentiment_words):
            if w not in all_summaries:
                all_summaries[w] = {}

            summary_v1, summary_v2 = get_contrastive_viewpoint_summary(w, n=50, k=100, model1=model1, model2=model2,
                                                                       mat_name='trans',
                                                                       dir_name_matrices=dir_name_matrices,
                                                                       save_dir=results_dir + 'summaries/',
                                                                       file_name='sentiment_keywords',
                                                                       viewpoint1_name='{}'.format(years[i] - 1),
                                                                       viewpoint2_name='{}'.format(years[i]),
                                                                       thresh=0.6)

            # 1) although we dont have sentences (POS is imp for entiment analysis, which can be determined in the context of a sentence)
            # but bcz of ocr errors two words are connected to each other, we separate them, this can be understood as a sentence
            # 2) should we determine pos/neg ? bcz positive in viewpoint 1 might be negative in viewpoint 2 ? rather; focus on objectivity !!!!!!!
            # 3) how to detemine sentiment of countries ? (THEY CARRY SENTIMENT) - like israeli-syrian, or israel, or palestine ?

            if str(years[i] - 1) not in all_summaries[w]:
                all_summaries[w][str(years[i] - 1)] = {}
                print('storing corrected summaries of {} for viewpoint {}'.format(w, str(years[i] - 1)))
                corrections = store_summary_corrections(summary_v1, save_dir=results_dir + 'summaries/sentiment_keywords/',
                                          filename='{}_{}'.format(word2idx[w], str(years[i] - 1)))
                all_summaries[w][str(years[i] - 1)] = corrections

            if str(years[i]) not in all_summaries[w]:
                all_summaries[w][str(years[i])] = {}
                print('storing corrected summaries of {} for viewpoint {}'.format(w, str(years[i])))
                corrections = store_summary_corrections(summary_v2, save_dir=results_dir + 'summaries/sentiment_keywords/',
                                          filename='{}_{}'.format(str(word2idx[w]), str(years[i])))
                all_summaries[w][str(years[i])] = corrections

            with open('all_summaries.pkl', 'wb') as f:
                pickle.dump(all_summaries, f)

        # ranks_comb, ranks_neigh, ranks_lin = get_ranks(stability_combined=stabilities_comb,
        #                                                stability_neighbors=stabilities_neigh,
        #                                                stability_linear=stabilities_lin)
        # # paired two tail t-test
        # perform_paired_t_test(ranks_comb, ranks_neigh, ranks_lin, save_dir=results_dir + 'significance/',
        #                       file_name=fig_name_general_prefix + '-' + str(yearsforfigs[i]))
        #
        # # delta of the ranks between neighbors and linear vs. combination
        # plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, ethnicities,
        #                        save_dir=results_dir + 'delta_ranks/', fig_name=fig_name_prefixes[i] + '_ethnicities')
        # plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, ideologies,
        #                        save_dir=results_dir + 'delta_ranks/', fig_name=fig_name_prefixes[i] + '_ideologies')
        # plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, political_parties,
        #                        save_dir=results_dir + 'delta_ranks/', fig_name=fig_name_prefixes[i] + '_political_parties')
        # plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, politicians,
        #                        save_dir=results_dir + 'delta_ranks/', fig_name=fig_name_prefixes[i] + '_politicians')
        # # plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, israeli_leaders,
        # #                        save_dir=results_dir + 'delta_ranks/', fig_name=fig_name_prefixes[i] + '_israeli_leaders')
        #
        # save_heads_tails_all(stabilities_comb=stabilities_comb, stabilities_neigh=stabilities_neigh,
        #                      stabilities_lin=stabilities_lin, n=50, verbose=False,
        #                      save_heads_tails=True,
        #                      save_dir=results_dir + 'heads_tails/',
        #                      file_name=fig_name_general_prefix + str(years[i]))

    # # heatmap of stability values for each word of interest
    # generate_stability_heatmap(ethnicities, stability_dicts_combined, stability_dicts_neighbor,
    #                            stability_dicts_linear,
    #                            years=yearsforfigs,
    #                            save_dir=results_dir + 'heatmap/', fig_name=fig_name_general_prefix + '_ethnicities')
    #
    # generate_stability_heatmap(ideologies, stability_dicts_combined, stability_dicts_neighbor,
    #                            stability_dicts_linear,
    #                            years=yearsforfigs,
    #                            save_dir=results_dir + 'heatmap/', fig_name=fig_name_general_prefix + '_ideologies')
    #
    # generate_stability_heatmap(political_parties, stability_dicts_combined, stability_dicts_neighbor,
    #                            stability_dicts_linear,
    #                            years=yearsforfigs,
    #                            save_dir=results_dir + 'heatmap/', fig_name=fig_name_general_prefix + '_political_parties')
    #
    # generate_stability_heatmap(politicians, stability_dicts_combined, stability_dicts_neighbor,
    #                            stability_dicts_linear,
    #                            years=yearsforfigs,
    #                            save_dir=results_dir + 'heatmap/', fig_name=fig_name_general_prefix + '_politicians')
    # # generate_stability_heatmap(israeli_leaders, stability_dicts_combined, stability_dicts_neighbor,
    # #                            stability_dicts_linear,
    # #                            years=yearsforfigs,
    # #                            save_dir=results_dir + 'heatmap/', fig_name=fig_name_general_prefix + '_israeli_leaders')
    #
    # # jaccard similarity between the tails of the stability dictionaries across years
    # plot_jaccard_similarity_tails(stability_dicts_combined,
    #                               stability_dicts_neighbor,
    #                               stability_dicts_linear,
    #                               n_sizes=list(range(10000, 110000, 10000)),
    #                               save_dir=results_dir, fig_name=fig_name_general_prefix)
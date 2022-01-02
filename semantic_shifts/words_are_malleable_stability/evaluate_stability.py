import pickle
import os
import matplotlib.pyplot as plt
from itertools import cycle
from bidi import algorithm as bidialg
import arabic_reshaper
from words_are_malleable import get_stability_combined_one_word
import fasttext
from scipy import stats
import numpy as np


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


def plot_jaccard_similarity_tails(stability_dicts_combined, stability_dicts_neighbor, n_sizes):
    """ get the jaccard similarity between the tails of the different stability
        approaches for different sizes of n. Ideally, because words in tails should be stable,
        they must be present in the tails of any corpus used.
    """
    jaccard_sims_comb, jaccard_sims_neigh = [], []
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

    lines = ["--", "-."]
    linecycler = cycle(lines)
    plt.figure()
    for i in range(len(lines)):
        if i == 0:
            plt.plot(list(n_sizes), jaccard_sims_comb, next(linecycler), label="Combination")
        else:
            plt.plot(list(n_sizes), jaccard_sims_neigh, next(linecycler), label="Neighbors-based")
    plt.legend()
    plt.xlabel('tail sizes')
    plt.ylabel('jaccard similarity')
    plt.xlim([0, n_sizes[-1]])
    plt.ylim([0, max(max(jaccard_sims_comb), max(jaccard_sims_neigh))])
    plt.savefig('jaccard-similarities.png')
    plt.close()


def plot_delta_ranks_words(ranks_comb, ranks_neigh, words):
    deltas = []
    words_decoded = []
    for w in words:
        dr = ranks_neigh[w] - ranks_comb[w] # get the delta rank
        deltas.append(dr) # add to list
        words_decoded.append(bidialg.get_display(arabic_reshaper.reshape(w)))
    plt.bar(words_decoded, deltas)
    plt.xticks(rotation=90)
    plt.ylabel(r'$\Delta$' + 'rank')
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    fig.tight_layout()
    plt.savefig('delta-ranks.png')
    plt.close()


def get_ranks(stability_combined, stability_neighbors):
    # sort the stabilities dictionary by increasing order of stabilities (items at the beginning
    # have low stability - items at the end have high stability)
    stability_combined = {k: v for k, v in sorted(stability_combined.items(), key=lambda item: item[1])}
    stability_neighbors = {k: v for k, v in sorted(stability_neighbors.items(), key=lambda item: item[1])}

    values_combined = list(stability_combined.values())
    values_neighbor = list(stability_neighbors.values())

    ranks_combined, ranks_neigh = [], []

    ranks_combined.append(1) # for the first value, its rank is 1
    ranks_neigh.append(1) # for the first value, its rank is 1

    # get the rankings per value for the combined
    rank = 1
    for i in range(1, len(values_combined[1:]) + 1):
        if round(values_combined[i], 5) == round(values_combined[i-1], 5):
            ranks_combined.append(rank)
        else:
            rank += 1
            ranks_combined.append(rank)
    print(len(ranks_combined) == len(values_combined))

    # get the rankings per value for the neighbors
    rank = 1
    for i in range(1, len(values_neighbor[1:]) + 1):
        if round(values_neighbor[i], 8) == round(values_neighbor[i-1], 8):
            ranks_neigh.append(rank)
        else:
            rank += 1
            ranks_neigh.append(rank)
    print(len(ranks_neigh) == len(values_neighbor))

    ranks_combined = dict(zip(list(stability_combined.keys()), ranks_combined))
    ranks_neigh = dict(zip(list(stability_neighbors.keys()), ranks_neigh))

    return ranks_combined, ranks_neigh


# cannot decide yet on summary because our words are oov and nearest
# neighs of oov are also oov, should we also get stability values
# for these ?

def get_contrastive_viewpoint_summary(w, n, k, model1, model2, mat_name, dir_name_matrices,
                                      viewpoint=1, thresh=0.6):
    """ get a contrastive viewpoint summary of a word of length n """
    summary = []
    count = 0
    if viewpoint == 1:
        nns = [n[1] for n in model1.get_nearest_neighbors(w, k)]
    else:
        nns = [n[1] for n in model2.get_nearest_neighbors(w, k)]
    for nn in nns:
        if count == n:
            break
        st = get_stability_combined_one_word(w=nn, model1=model1, model2=model2,mat_name=mat_name,
                                        dir_name_matrices=dir_name_matrices)

        if st <= thresh:
            summary.append((st, nn))
            count += 1
    for s in summary:
        print(s)


def perform_paired_t_test(ranks_comb, ranks_neigh):
    result = stats.ttest_rel(list(ranks_comb.values()), list(ranks_neigh.values()))
    print('avg rank combined: {}'.format(np.mean(list(ranks_comb.values()))))
    print('avg rank neighbors: {}'.format(np.mean(list(ranks_neigh.values()))))
    if result[1] < 0.05:
        print(result)
        print('accept alternative hypothesis')
    else:
        print(result)
        print('accept alternative hypothesis')


if __name__ == '__main__':
    # path1 = 'E:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/'
    # path2 = 'E:/fasttext_embeddings/assafir/'
    # model1 = fasttext.load_model(os.path.join(path1, '2007.bin'))
    # model2 = fasttext.load_model(os.path.join(path2, '2007.bin'))
    with open('../input/keywords.txt', 'r', encoding='utf-8') as f:
        words = f.readlines()
    dir_name_matrices = 'E:/fasttext_embeddings/results/nahar_2007_assafir_2007/linear_numsteps70000/matrices/'
    words = [w[:-1] for w in words if '\n' in w]
    print(words)
    # for w in words:
    #     print('{}:'.format(w))
    #     get_contrastive_viewpoint_summary(w, n=25, k=100, model1=model1, model2=model2,
    #                                       mat_name='trans', dir_name_matrices=dir_name_matrices,
    #                                       viewpoint=1, thresh=0.6)
    #     print('========================================================================')


    paths = [
             # 'E:/fasttext_embeddings/results/nahar_1982_assafir_1982/t1k100/',
             # 'E:/fasttext_embeddings/results/nahar_1982_assafir_1982/t1k200/',
             # 'E:/fasttext_embeddings/results/nahar_1982_assafir_1982/t1k300/',
            # 'E:/fasttext_embeddings/results/nahar_1982_assafir_1982/linear_numsteps70000/matrices/',

             # 'E:/fasttext_embeddings/results/nahar_2006_assafir_2006/t1k100/',
             # 'E:/fasttext_embeddings/results/nahar_2006_assafir_2006/t1k200/',
             # 'E:/fasttext_embeddings/results/nahar_2006_assafir_2006/t1k300/',
             # 'E:/fasttext_embeddings/results/nahar_2006_assafir_2006/linear_numsteps70000/matrices/',
             #
             'E:/fasttext_embeddings/results/nahar_2007_assafir_2007/t1k100/',
             # 'E:/fasttext_embeddings/results/nahar_2007_assafir_2007/t1k200/',
             # 'E:/fasttext_embeddings/results/nahar_2007_assafir_2007/t1k300/',
             # 'E:/fasttext_embeddings/results/nahar_2007_assafir_2007/linear_numsteps70000/matrices/',
             ]

    stability_dicts_combined = []
    stability_dicts_neighbor = []
    for path in paths:
        dict_combined = os.path.join(path, 'stabilities_combined.pkl')
        dict_neighbor = os.path.join(path, 'stabilities_neighbor.pkl')
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

        ranks_comb, ranks_neigh = get_ranks(stability_combined=stabilities_comb, stability_neighbors=stabilities_neigh)
        perform_paired_t_test(ranks_comb, ranks_neigh)
        plot_delta_ranks_words(ranks_comb, ranks_neigh, words)
        break
    # plot_jaccard_similarity_tails(stability_dicts_combined, stability_dicts_neighbor, n_sizes=list(range(0, 110000, 10000)))
    # print('')
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


def generate_stability_heatmap(words, stability_dicts_combined, stability_dicts_neighbor,
                               stability_dicts_linear,
                               years, save_dir, fig_name):
    yticks = [bidialg.get_display(arabic_reshaper.reshape(w)) for w in words]
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        data = []
        for w in words:
            stab_vals = []
            if i == 0:
                for j in range(len(years)):
                    stab_vals.append(stability_dicts_combined[j][w])
                data.append(stab_vals)
            elif i == 1:
                for j in range(len(years)):
                    stab_vals.append(stability_dicts_neighbor[j][w])
                data.append(stab_vals)
            else:
                for j in range(len(years)):
                    stab_vals.append(stability_dicts_linear[j][w])
                data.append(stab_vals)
        data = np.array(data)
        sns.heatmap(data, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=False if i < 2 else True,
                    ax=ax[i])
        ax[i].set_xlabel("Combined" if i == 0 else "Neighbor-based" if i == 1 else "Linear-Mapping")
        ax[i].set_xticklabels(years, rotation=90)
    plt.tight_layout()
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
    plt.ylim([0, max(max(jaccard_sims_comb), max(jaccard_sims_neigh), max(jaccard_sims_lin))])
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '_jaccard_similarities.png'))
    plt.close()


def plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, words,
                           save_dir, fig_name):
    deltas_neigh, deltas_lin = [], []
    words_decoded = []
    for w in words:
        drneigh = ranks_neigh[w] - ranks_comb[w] # get the delta rank
        drlin = ranks_lin[w] - ranks_comb[w]     # get the delta rank
        deltas_neigh.append(drneigh) # add to list
        deltas_lin.append(drlin)     # add to list
        words_decoded.append(bidialg.get_display(arabic_reshaper.reshape(w))) # decode arabic word to make it appear in matplotlib
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

    ranks_combined.append(1) # for the first value, its rank is 1
    ranks_neigh.append(1) # for the first value, its rank is 1
    ranks_lin.append(1)

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
                                      save_dir, file_name, viewpoint=1, thresh=0.6):
    """ get a contrastive viewpoint summary of a word of length n. For a certain
        word, we get its top k nearest neighbors. Then for each nearest neighbor, we add it
        into the summary if its stability is less than a certain threshold.
    """
    summary = []
    count = 0
    if viewpoint == 1:
        nns = [n[1] for n in model1.get_nearest_neighbors(w, k)]
    else:
        nns = [n[1] for n in model2.get_nearest_neighbors(w, k)]
    for nn in nns:
        if count == n:
            break
        st = get_stability_combined_one_word(w=nn, model1=model1, model2=model2, mat_name=mat_name,
                                        dir_name_matrices=dir_name_matrices)

        if st <= thresh:
            summary.append((st, nn))
            count += 1
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}_{}.txt'.format(file_name, 1)), encoding='utf-8') as f:
        f.writelines('w: {} - viewpoint {}'.format(w, viewpoint))
        for i, s in enumerate(summary):
            if i%5 != 0 or i == 0:
                f.write(s + ", ")
            else:
                f.write(s + "\n")
    f.close()


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
    # model1 = fasttext.load_model(os.path.join(path1, '2006.bin'))
    # model2 = fasttext.load_model(os.path.join(path2, '2006.bin'))

    with open('../input/keywords.txt', 'r', encoding='utf-8') as f:
        words = f.readlines()
    dir_name_matrices = 'E:/fasttext_embeddings/results/nahar_2007_assafir_2007/linear_numsteps70000/matrices/'
    words = [w[:-1] for w in words if '\n' in w]
    print(words)

    # for w in words:
    #     get_contrastive_viewpoint_summary(w, n=25, k=100, model1=model1, model2=model2,
    #                                       mat_name='trans', dir_name_matrices=dir_name_matrices,
    #                                       viewpoint=1, thresh=0.6)
    #     get_contrastive_viewpoint_summary(w, n=25, k=100, model1=model1, model2=model2,
    #                                       mat_name='trans', dir_name_matrices=dir_name_matrices,
    #                                       viewpoint=2, thresh=0.6)
    #     print('///////////////////////////////////////////////////////////////////')

    fig_name_prefixes = [
        'nahar_2000_assafir_2000',
        'nahar_2001_assafir_2001',
        'nahar_2002_assafir_2002',

        'nahar_2006_assafir_2006',
        'nahar_2007_assafir_2007',
        'nahar_2008_assafir_2008'
    ]
    fig_name_general_prefix = 'nahar_assafir'
    paths = [
        'E:/fasttext_embeddings/results/nahar_2000_assafir_2000/t1k100/',
        'E:/fasttext_embeddings/results/nahar_2001_assafir_2001/t1k100/',
        'E:/fasttext_embeddings/results/nahar_2002_assafir_2002/t1k100/',

        'E:/fasttext_embeddings/results/nahar_2006_assafir_2006/t1k100/',
        'E:/fasttext_embeddings/results/nahar_2007_assafir_2007/t1k100/',
        'E:/fasttext_embeddings/results/nahar_2008_assafir_2008/t1k100/',
    ]

    stability_dicts_combined = []
    stability_dicts_neighbor = []
    stability_dicts_linear = []
    results_dir = 'output/'

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
            with open(dict_neighbor, 'rb') as handle:
                stabilities_lin = pickle.load(handle)
                stability_dicts_linear.append(stabilities_lin)
            print('================================================================')

        ranks_comb, ranks_neigh, ranks_lin = get_ranks(stability_combined=stabilities_comb,
                                                       stability_neighbors=stabilities_neigh,
                                                       stability_linear=stabilities_lin)
        # paired two tail t-test
        perform_paired_t_test(ranks_comb, ranks_neigh)

        # delta of the ranks between neighbors and linear vs. combination
        plot_delta_ranks_words(ranks_comb, ranks_neigh, ranks_lin, words,
                               save_dir=results_dir, fig_name=fig_name_prefixes[i])

    # heatmap of stability values for each word of interest
    generate_stability_heatmap(words, stability_dicts_combined, stability_dicts_neighbor,
                               stability_dicts_linear,
                               years=[2000, 2001, 2002, 2006, 2007, 2008],
                               save_dir=results_dir, fig_name=fig_name_general_prefix)

    # jaccard similarity between the tails of the stability dictionaries across years
    plot_jaccard_similarity_tails(stability_dicts_combined,
                                  stability_dicts_neighbor,
                                  stability_dicts_linear,
                                  n_sizes=list(range(10000, 110000, 10000)),
                                  save_dir=results_dir, fig_name=fig_name_general_prefix)
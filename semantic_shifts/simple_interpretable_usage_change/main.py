from sklearn.manifold import TSNE
import argparse
import matplotlib.pyplot as plt
import fasttext
import numpy as np
import os, pickle
from nltk.util import ngrams


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_intersection_with_ocr_errors_ngram(w, k, models):
    common = set()
    neighs1 = [n[1] for n in models[0].get_nearest_neighbors(w, k)]
    neighs2 = [n[1] for n in models[1].get_nearest_neighbors(w, k)]
    for nn1 in neighs1:
        for nn2 in neighs2:
            overlaps = []
            # print('nn1: {} / nn2: {}'.format(nn1, nn2))
            for n in range(1, 5):
                if len(nn1) >= n and len(nn2) >= n:
                    # get the ngrams of each
                    ngrams_nn1 = set(list(ngrams(nn1, n)))
                    ngrams_nn2 = set(list(ngrams(nn2, n)))

                    # now calculate n-gram overlap:
                    inter = ngrams_nn1.intersection(ngrams_nn2)
                    un = ngrams_nn1.union(ngrams_nn2)
                    if len(inter) != 0 and len(un) != 0:
                        ngram_overlap = len(inter) / len(un)
                        overlaps.append(ngram_overlap)
            results = [i for i in range(len(overlaps)) if overlaps[i] >= 0.8]
            if len(results) >= 1:
                # print('{} and {} are the same words: {}-grams >= 0.8'.format(nn1, nn2, results))
                common.add(nn1)
                common.add(nn2)
                break

    return common, neighs1, neighs2

# def get_intersection_with_ocr_errors(w, k, models):
#     common = set()
#     found = False
#     neighs1 = [n[1] for n in models[0].get_nearest_neighbors(w, k)]
#     neighs2 = [n[1] for n in models[1].get_nearest_neighbors(w, k)]
#
#     for nn1 in neighs1:
#         for nn2 in neighs2:
#             # maxlen = max(len(nn1), len(nn2))
#             # if there is a huge intersection between the characters of both words
#             # if maxlen - len(''.join(set(nn1).intersection(nn2))) <= 2:
#             ngrams_nn1, ngrams_nn2 = [], []
#             for i in range(len(nn1)):
#                 ngrams_nn1.append(nn1[0:i])
#                 ngrams_nn1.append(nn1[i:])
#             for i in range(len(nn2)):
#                 ngrams_nn2.append(nn2[0:i])
#                 ngrams_nn2.append(nn2[i:])
#             # get the intersection
#             cmn = set(ngrams_nn1).intersection(set(ngrams_nn2))
#             # sort by decreasing order of length
#             cmn_sorted = sorted(list(cmn), key=lambda x: (-len(x), x))
#             # if there exist a word in the intersection that is less than the original word by 40% max, then add it
#             for cw in cmn_sorted:
#                 if abs((len(nn1) - len(cw))) / len(nn1) <= 0.2 and abs((len(nn2) - len(cw))) / len(nn2) <= 0.2:
#                     common.add(nn1)
#                     common.add(nn2)
#                     # print('{} is subset of {}'.format(cw, (nn1, nn2)))
#                     found = True
#                     break
#
#             if found:
#                 found = False
#                 break
#     return common, neighs1, neighs2


# set(out[1] for out in model1.get_nearest_neighbors(w, args.k))
def tsne_plot(val1, val2):
    # visualize the top-k neighbors of each interesting word in both spaces

    # run through all the words of interest
    words_file = args.words
    if words_file.endswith('.txt'):
        with open(words_file, 'r') as f:
            words = f.readlines()
    elif words_file.endswith('.pkl'):
        with open(words_file, 'rb', encoding='utf-8') as f:
            words = pickle.load(f)
    else:
        words = words_file.strip().split(",")

    for int_word in words:
        # ensure the word is in both spaces
        if int_word in vocab[val1] and int_word in vocab[val2]:
            # identify the top-k neighbors
            neighbors_a = set(out[1] for out in model1.get_nearest_neighbors(int_word, args.k))
            neighbors_b = set(out[1] for out in model2.get_nearest_neighbors(int_word, args.k))
            total_neighbors = neighbors_a.union(neighbors_b)
            intersection_neighbors = get_intersection_with_ocr_errors(neighs1=neighbors_a, neighs2=neighbors_b)

            # identify neighbors which occur in a specific space and common space
            neighbor2color = {int_word: 'green'}  # coloring code - green for word of interest
            common, na, nb = [], [], []  # na, nb contains neighbors which are in specific space
            for neighbor in total_neighbors:
                if neighbor in intersection_neighbors:
                    neighbor2color[neighbor] = 'purple'  # coloring code - purple for neighbors in common space
                    common.append(neighbor)
                elif neighbor in neighbors_a:
                    neighbor2color[neighbor] = 'cyan'  # coloring code - cyan for neighbors in space 'a'
                    na.append(neighbor)
                else:
                    neighbor2color[neighbor] = 'violet'  # coloring code - violet for neighbors in space 'b'
                    nb.append(neighbor)

            # run over each space
            for val in [val1, val2]:
                # construct embedding matrix (tsne input) for neighboring words
                X, wname, colors = [], [], []
                # X.append(wv[val][w2i[val][int_word]]) # as if wv[val][index of word] i.e. get the vector of the word
                if val == val1:
                    if ' ' in int_word:
                        X.append(model1.get_sentence_vector(int_word))
                    else:
                        X.append(model1.get_word_vector(int_word))
                else:
                    if ' ' in int_word:
                        X.append(model2.get_sentence_vector(int_word))
                    else:
                        X.append(model2.get_word_vector(int_word))
                wname.append(int_word)
                colors.append('green')
                for word in sorted(total_neighbors):
                    if val == val1:
                        if word in vocab[val]:
                            X.append(model1.get_word_vector(int_word))
                            wname.append(word)
                            colors.append(neighbor2color[word])
                    else:
                        if word in vocab[val]:
                            X.append(model2.get_word_vector(int_word))
                            wname.append(word)
                            colors.append(neighbor2color[word])
                X = np.array(X, dtype=np.float)
                # perform tsne
                embeddings = TSNE(n_components=2, verbose=2, perplexity=30, n_iter=1000).fit_transform(X)
                # make tsne plot
                xx, yy = embeddings[:, 0], embeddings[:, 1]
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.scatter(xx, yy, c=colors)
                plt.title('t-SNE for word %s in space %s' % (int_word, val))
                for wi, word in enumerate(wname):
                    if wi == 0:
                        plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=20)
                    if wi % 1 == 0:
                        plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=10)
                mkdir(args.out_dir)
                fig.savefig(args.out_dir + "sp%s_w%s.pdf" % (val, int_word), bbox_inches='tight')
        else:
            print('skipping word %s' % int_word)


def read_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.readlines()
    words = [w[:-1] for w in words if '\n' in w]
    words = [w for w in words if w.strip() != '']
    words = [w.strip() for w in words]
    return words


def save_stability(stabilityovertime, save_dir):
    """ mapping of word vs viewpoint (time point) vs score (stability) """
    mkdir(save_dir)
    with open(os.path.join(save_dir, 'stability_dict.pickle'), 'wb') as handle:
        pickle.dump(stabilityovertime, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_summary(summaryovertime, save_dir):
    """ mapping of word vs viewpoint (time point) vs summary (set of nearest neighors from each view point) """
    mkdir(save_dir)
    with open(os.path.join(save_dir, 'summary_dict.pickle'), 'wb') as handle:
        pickle.dump(summaryovertime, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path1", default='E:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/', help="")
    parser.add_argument("--models_path2", default='E:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/', help="")
    parser.add_argument('--keywords_path', default='../words_are_malleable_stability/from_DrFatima/words_threshold.txt')
    parser.add_argument("--mode", default="d-nahar", help="mode: \'d-archivename\' for diachronic, \'s\' for synchronic")
    parser.add_argument("--out_dir", default='evaluate_stability_gonen/', help="output directory to save results in")
    parser.add_argument("--n", type=int, default=10, help="n for the length ofthe summary (sumber of nearest neighbors for contrastive viewpoint summary - NOT included in calculation of the intersection)")
    parser.add_argument("--k", type=int, default=100, help="k for the length of the intersection - used for getting score(stability)")
    args = parser.parse_args()

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
            'years': [[y - 1, y] for y in list(range(1983, 2012))],
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
    years = dict_years[mode]['years']
    stabilities_over_time = {}
    summaries_over_time = {}
    results_dir = '{}/{}/'.format(args.out_dir, mode)
    k = args.k # the k here is the length of the intersection for finding the score
    n = args.n # the n here is the length of the summar for conrastive viewpoint summarization (not included in computing the intersection)

    for i, path in enumerate(years):
        models2load = dict_years[mode]['models'][i]
        viewpoints = dict_years[mode]['viewpoints'][i]
        years2load = dict_years[mode]['years'][i]
        time_point = dict_years[mode]['time_points'][i]

        path1 = args.models_path1
        path2 = args.models_path2

        # for sentiment
        sentiment_words = read_keywords(args.keywords_path)

        models = []  # to store loaded models inside an array to pass to the get_summaries method
        model1 = fasttext.load_model(os.path.join(path1, '{}'.format(models2load[0])))
        model2 = fasttext.load_model(os.path.join(path2, '{}'.format(models2load[1])))

        models.append(model1)
        models.append(model2)

        for w in sentiment_words:
            viewpoints_str = '_'.join(viewpoints)
            if w not in stabilities_over_time:
                stabilities_over_time[w] = {}
            if w not in summaries_over_time:
                summaries_over_time[w] = {}

            if viewpoints_str not in stabilities_over_time[w]:
                stabilities_over_time[w][viewpoints_str] = {}
            if viewpoints_str not in summaries_over_time[w]:
                summaries_over_time[w][viewpoints_str] = {}

            # viewpoints_str = '_'.join(viewpoints)
            # get the intersection along with the nearest neighbors from each embedding space
            # intersection, nn1, nn2 = get_intersection_with_ocr_errors(w=w, k=k, models=models)
            intersection, nn1, nn2 = get_intersection_with_ocr_errors_ngram(w=w, k=k, models=models)

            stabilities_over_time[w][viewpoints_str]['stability'] = len(list(intersection))
            summaries_over_time[w][viewpoints_str][viewpoints[0]] = nn1[:n]
            summaries_over_time[w][viewpoints_str][viewpoints[1]] = nn2[:n]

            print('stability of word {} in viewpoint {}: {}'.format(w, viewpoints_str, len(list(intersection))))

        save_stability(stabilityovertime=stabilities_over_time, save_dir=results_dir)
        save_summary(summaryovertime=summaries_over_time, save_dir=results_dir)

    # model1 = fasttext.load_model(args.embed_a)
    # model2 = fasttext.load_model(args.embed_b)
    # vocab = {}
    # vocab[val1] = model1.words
    # vocab[val2] = model2.words
    #
    # tsne_plot(val1, val2)

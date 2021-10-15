"""
Code inspired from https://github.com/gonenhila/usage_change/tree/master/source
"""
import os.path
import fasttext
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_fasttext import get_intersection_with_ocr_errors_ngram
from bidi import algorithm as bidialg
import arabic_reshaper


def tsne_plot(val1, val2):
    # visualize the top-k neighbors of each interesting word in both spaces

    # run through all the words of interest
    for int_word in words:
        # ensure the word is in both spaces
        neighbors_a = set([out[1] for out in model1.get_nearest_neighbors(int_word, args.k)])
        neighbors_b = set([out[1] for out in model2.get_nearest_neighbors(int_word, args.k)])

        total_neighbors = neighbors_a.union(neighbors_b)
        intersection_neighbors = get_intersection_with_ocr_errors_ngram(neighbors_a, neighbors_b)
        print(intersection_neighbors)
        # neighbours_a_new = neighbors_a.difference(intersection_neighbors)
        # neighbours_b_new = neighbors_b.difference(intersection_neighbors)
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
            if val == val1:
                X.append(model1.get_word_vector(int_word))
            else:
                X.append(model2.get_word_vector(int_word))
            wname.append(bidialg.get_display(arabic_reshaper.reshape(int_word)))
            colors.append('green')
            if val == val1:
                for word in sorted(total_neighbors):
                    # if word in neighbors_a:
                    X.append(model1.get_word_vector(word))
                    wname.append(bidialg.get_display(arabic_reshaper.reshape(word)))
                    colors.append(neighbor2color[word])
            else:
                for word in sorted(total_neighbors):
                    # if word in neighbors_b:
                    X.append(model2.get_word_vector(word))
                    wname.append(bidialg.get_display(arabic_reshaper.reshape(word)))
                    colors.append(neighbor2color[word])

            X = np.array(X, dtype=np.float)
            # perform tsne
            embeddings = TSNE(n_components=2, verbose=2, perplexity=30, n_iter=1000).fit_transform(X)
            # make tsne plot
            xx, yy = embeddings[:, 0], embeddings[:, 1]
            ax = plt.subplot(111)
            ax.scatter(xx, yy, c=colors)
            plt.title('t-SNE for word %s in space %s' % (bidialg.get_display(arabic_reshaper.reshape(int_word)), val))
            for wi, word in enumerate(wname):
                if wi == 0:
                    plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=20)
                if wi % 1 == 0:
                    plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=10)
            if not os.path.exists(args.out_topk):
                os.makedirs(args.out_topk)
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.savefig(args.out_topk + "sp%s_w%s" % (val, int_word), bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_a",
                        default='../Train_Word_Embedidng/fasttext/nahar/SGNS/'
                                'ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin',
                        help="prefix for embedding for split a")
    parser.add_argument("--embed_b",
                        default='../Train_Word_Embedidng/fasttext/assafir/SGNS/'
                                'ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin',
                        help="prefix for embedding for split b")
    parser.add_argument("--data_a", default='../Train_Word_Embedidng/fasttext/data/nahar/2006.txt',
                        help="name of tokenized data file for split a")
    parser.add_argument("--data_b", default='../Train_Word_Embedidng/fasttext/data/assafir/2006.txt',
                        help="name of tokenized data file for split b")
    parser.add_argument("--name_split_a", default='nahar_2006', help="short name for split a")
    parser.add_argument("--name_split_b", default='assafir_2006', help="short name for split b")
    parser.add_argument("--words", default="words_1975_1990.txt", help="path to words file")
    parser.add_argument("--out_topk",
                        default='/tmp/result',
                        help="prefix for the output files for topk and latex table")
    parser.add_argument("--freq_thr", type=float, default=0.00001, help="frequency threshold")
    parser.add_argument("--min_count", type=int, default=200, help="min appearances of a word")
    parser.add_argument("--k", type=int, default=10000, help="k of k-NN to use")

    args = parser.parse_args()

    print('Embedding 1: {}'.format(args.embed_a))
    print('Embedding 2: {}'.format(args.embed_b))
    print('data 1: {}'.format(args.data_a))
    print('data 2: {}'.format(args.data_b))
    print('split name 1: {}'.format(args.name_split_a))
    print('split name 2: {}'.format(args.name_split_b))
    print('words file: {}'.format(args.words))

    model1 = fasttext.load_model(args.embed_a)
    model2 = fasttext.load_model(args.embed_b)

    val1, val2 = args.name_split_a, args.name_split_b

    with open(args.words) as file:
        lines = file.readlines()
        words = [line.rstrip() for line in lines]
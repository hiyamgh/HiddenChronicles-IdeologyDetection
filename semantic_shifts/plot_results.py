from sklearn.manifold import TSNE
import argparse
import matplotlib.pyplot as plt
import fasttext
import numpy as np
import os


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_intersection_with_ocr_errors(neighs1, neighs2):
    common = set()
    found = False
    for nn1 in neighs1:
        for nn2 in neighs2:
            maxlen = max(len(nn1), len(nn2))
            # if there is a huge intersection between the characters of both words
            if maxlen - len(''.join(set(nn1).intersection(nn2))) <= 2:
                ngrams_nn1, ngrams_nn2 = [], []
                for i in range(len(nn1)):
                    ngrams_nn1.append(nn1[0:i])
                    ngrams_nn1.append(nn1[i:])
                for i in range(len(nn2)):
                    ngrams_nn2.append(nn2[0:i])
                    ngrams_nn2.append(nn2[i:])
                # get the intersection
                cmn = set(ngrams_nn1).intersection(set(ngrams_nn2))
                # sort by decreasing order of length
                cmn_sorted = sorted(list(cmn), key=lambda x: (-len(x), x))
                # get original word for comparison
                original = nn1 if len(nn1) >= len(nn2) else nn2
                # if there exist a word in the intersection that is less than the original word by 40% max, then add it
                for cw in cmn_sorted:
                    if abs((len(nn1) - len(cw))) / len(nn1) <= 0.2 and abs((len(nn2) - len(cw))) / len(nn2) <= 0.2:
                        # common.add(nn1 if len(nn1) >= len(nn2) else nn2)
                        # common.add(cw)
                        # print('{} is subset of {}, added {}'.format(cw, original, (nn1, nn2)))
                        # break
                        common.add(nn1)
                        common.add(nn2)
                        print('{} is subset of {}'.format(cw, (nn1, nn2)))
                        found = True
                        break

            if found:
                found = False
                break
    return common


# set(out[1] for out in model1.get_nearest_neighbors(w, args.k))
def tsne_plot(val1, val2):
    # visualize the top-k neighbors of each interesting word in both spaces

    # run through all the words of interest
    for int_word in args.words.strip().split(","):
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
                    X.append(model1.get_word_vector(int_word))
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

    parser.add_argument("--name_split_a",
                        default='nahar_2006',
                        help="short name for split a")
    parser.add_argument("--name_split_b",
                        default='assafir_2006',
                        help="short name for split b")
    parser.add_argument("--out_dir",
                        default='/tmp/result',
                        help="prefix for the output files for topk and latex table")
    parser.add_argument("--words", default='الاسرائيلي,الأاميركي,الاتحادالسوفياتي', help="interesting words to plot in csv format")
    parser.add_argument("--k", type=int, default=10, help="k of k-NN to use")
    args = parser.parse_args()

    val1, val2 = args.name_split_a, args.name_split_b

    model1 = fasttext.load_model(args.embed_a)
    model2 = fasttext.load_model(args.embed_b)
    vocab = {}
    vocab[val1] = model1.words
    vocab[val2] = model2.words

    tsne_plot(val1, val2)

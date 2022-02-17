import fasttext
import argparse
import operator
from collections import defaultdict
from tqdm import tqdm
import pickle
from nltk.util import ngrams
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def extract_freqs(filename, vocab):
    ''' extract word raw frequencies and normalized frequencies '''
    print('extracting freqs %s' % filename)
    count = defaultdict(int)
    with open(filename, 'r') as f:
        for l in f:
            for w in l.strip().split():
                count[w] += 1

    # consider only words in the vocabulary
    count_vocab = defaultdict(int)
    for w in vocab:
        if w in count:
            count_vocab[w] = count[w]

    # normalized frequencies
    tot = sum([count_vocab[item] for item in count_vocab])
    freq_norm = defaultdict(int)
    for w in count_vocab:
        freq_norm[w] = count_vocab[w] / float(tot)

    # top-frequent
    top_freq = defaultdict(int)
    sorted_words = [x[0] for x in sorted(count_vocab.items(), key=operator.itemgetter(1))]
    cutoff = len(sorted_words) / float(20)
    top_freq_words = sorted_words[int(4 * cutoff):-200]  # -int(cutoff)]
    for w in top_freq_words:
        top_freq[w] = count[w]

    print('done')
    return freq_norm, count_vocab, top_freq


def word_grams(word, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(word, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

#
# def tsne_plot(val1, val2):
#     # visualize the top-k neighbors of each interesting word in both spaces
#
#     # run through all the words of interest
#     for int_word in words.strip().split(","):
#         # ensure the word is in both spaces
#         if int_word in vocab[val1] and int_word in vocab[val2]:
#             # identify the top-k neighbors
#             neighbors_a = set(topK(int_word, val1, k, None, 100))
#             neighbors_b = set(topK(int_word, val2, k, None, 100))
#             total_neighbors = neighbors_a.union(neighbors_b)
#             # identify neighbors which occur in a specific space and common space
#             neighbor2color = {int_word: 'green'}  # coloring code - green for word of interest
#             common, na, nb = [], [], []  # na, nb contains neighbors which are in specific space
#             for neighbor in total_neighbors:
#                 if neighbor in neighbors_a and neighbor in neighbors_b:
#                     neighbor2color[neighbor] = 'purple'  # coloring code - purple for neighbors in common space
#                     common.append(neighbor)
#                 elif neighbor in neighbors_a:
#                     neighbor2color[neighbor] = 'cyan'  # coloring code - cyan for neighbors in space 'a'
#                     na.append(neighbor)
#                 else:
#                     neighbor2color[neighbor] = 'violet'  # coloring code - violet for neighbors in space 'b'
#                     nb.append(neighbor)
#
#             # run over each space
#             for val in [val1, val2]:
#                 # construct embedding matrix (tsne input) for neighboring words
#                 X, wname, colors = [], [], []
#                 X.append(wv[val][w2i[val][int_word]])
#                 wname.append(int_word)
#                 colors.append('green')
#                 for word in sorted(total_neighbors):
#                     if word in w2i[val]:
#                         X.append(wv[val][w2i[val][word]])
#                         wname.append(word)
#                         colors.append(neighbor2color[word])
#                 X = np.array(X, dtype=np.float)
#                 # perform tsne
#                 embeddings = TSNE(n_components=2, verbose=2, perplexity=30, n_iter=1000).fit_transform(X)
#                 # make tsne plot
#                 xx, yy = embeddings[:, 0], embeddings[:, 1]
#                 fig = plt.figure()
#                 ax = plt.subplot(111)
#                 ax.scatter(xx, yy, c=colors)
#                 plt.title('t-SNE for word %s in space %s' % (int_word, val))
#                 for wi, word in enumerate(wname):
#                     if wi == 0:
#                         plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=20)
#                     if wi % 1 == 0:
#                         plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=10)
#                 fig.savefig(out_dir + "sp%s_w%s.pdf" % (val, int_word), bbox_inches='tight')
#         else:
#             print('skipping word %s' % int_word)


def get_intersection_with_ocr_errors_ngram(neighs1, neighs2):
    common = set()
    for nn1 in neighs1:
        for nn2 in neighs2:
            overlaps = []
            print('nn1: {} / nn2: {}'.format(nn1, nn2))
            for n in range(1, 5):
                if len(nn1) >= n and len(nn2) >= n:
                    print('yes')
                    ngrams_nn1 = set(list(ngrams(nn1, n)))
                    ngrams_nn2 = set(list(ngrams(nn2, n)))
                    print('ngram1: {}'.format(ngrams_nn1))
                    print('ngram2: {}'.format(ngrams_nn2))
                    # now calculate n-gram overlap:
                    inter = ngrams_nn1.intersection(ngrams_nn2)
                    un = ngrams_nn1.union(ngrams_nn2)
                    # dif = un.difference(inter)
                    print('intersection: {}'.format(inter))
                    print('union: {}'.format(un))
                    # print('difference: {}'.format(dif))
                    if len(inter) != 0 and len(un) != 0:
                        ngram_overlap = len(inter) / len(un)
                        overlaps.append(ngram_overlap)
            results = [val for val in overlaps if val >= 0.8]
            if len(results) >= 1:
                # get the 'n':
                # grams = [overlaps[i] for i in range(len(overlaps)) if overlaps[i] >= 0.8]
                print('{} and {} are the same words: {}-grams >= 0.8'.format(nn1, nn2, results))
                common.add(nn1)
                common.add(nn2)
                break

    return common


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


def NN_scores():
    # rank all the words based on our method

    # compute nearest neighbors overlap for all the common words
    nn_scores = []
    pbar = tqdm(total=len(vocab[val1]))
    for i, w in enumerate(vocab[val1]):
        # if w not in s_words and w in freq1 and w in freq2 and count1[w] > MIN_COUNT and count2[w] > MIN_COUNT:
        # if w in freq1 and w in freq2 and count1[w] > MIN_COUNT and count2[w] > MIN_COUNT:
            # neighbors_bef = set(topK(w, space1, k, count1, 100))
            # neighbors_aft = set(topK(w, sapce2, k, count2, 100))

        if w in vocab[val1] and w in vocab[val2]:
            neighbors_bef = set(out[1] for out in model1.get_nearest_neighbors(w, k))
            neighbors_aft = set(out[1] for out in model2.get_nearest_neighbors(w, k))
            # common = get_intersection_with_ocr_errors(neighs1=neighbors_bef, neighs2=neighbors_aft)
            common = get_intersection_with_ocr_errors_ngram(neighs1=neighbors_bef, neighs2=neighbors_aft)
            nn_scores.append((len(common), w))

        if i % 10 == 0:
            pbar.update(10)
    pbar.close()
    print('len of ranking', len(nn_scores))

    # rank these words
    nn_scores_sorted = sorted(nn_scores)
    for t in nn_scores[:50]:
        print(t)

    with open('nn_scores_{}_{}.pkl'.format(val1, val2), 'wb') as f:
        pickle.dump(nn_scores_sorted, f)
    return nn_scores_sorted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    embed_a = 'E:/nahar_2000_2006.bin'
    embed_b = 'E:/assafir_2000_2006.bin'
    data_a = 'E:/nahar_2000-2006.txt'
    data_b = 'E:/assafir_2000-2006.txt'
    name_split_a = 'nahar_2000_2006'
    name_split_b = 'assafir_2000_2006'
    out_topk = 'E:/results/'
    k = 200
    # parser.add_argument("--embed_a",
    #                     default='../Train_Word_Embedidng/fasttext/nahar/SGNS/'
    #                             'ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin',
    #                     help="prefix for embedding for split a")
    # parser.add_argument("--embed_b",
    #                     default='../Train_Word_Embedidng/fasttext/assafir/SGNS/'
    #                             'ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin',
    #                     help="prefix for embedding for split b")
    # parser.add_argument("--data_a",
    #                     default='../Train_Word_Embedidng/fasttext/data/nahar/2006.txt',
    #                     help="name of tokenized data file for split a")
    # parser.add_argument("--data_b",
    #                     default='../Train_Word_Embedidng/fasttext/data/assafir/2006.txt',
    #                     help="name of tokenized data file for split b")
    # parser.add_argument("--name_split_a",
    #                     default='nahar_2006',
    #                     help="short name for split a")
    # parser.add_argument("--name_split_b",
    #                     default='assafir_2006',
    #                     help="short name for split b")
    # parser.add_argument("--out_topk",
    #                     default='/tmp/result',
    #                     help="prefix for the output files for topk and latex table")
    # parser.add_argument("--freq_thr", type=float, default=0.00001, help="frequency threshold")
    # parser.add_argument("--min_count", type=int, default=200, help="min appearances of a word")
    # parser.add_argument("--k", type=int, default=10000, help="k of k-NN to use")

    # args = parser.parse_args()

    print('Embedding 1: {}'.format(embed_a))
    print('Embedding 2: {}'.format(embed_b))
    print('data 1: {}'.format(data_a))
    print('data 2: {}'.format(data_b))
    print('split name 1: {}'.format(name_split_a))
    print('split name 2: {}'.format(name_split_b))

    model1 = fasttext.load_model(embed_a)
    model2 = fasttext.load_model(embed_b)

    val1, val2 = name_split_a, name_split_b

    vocab = {}
    vocab[val1] = model1.words
    vocab[val2] = model2.words

    # freq_norm_val1, count_vocab_val1, top_freq_val1 = extract_freqs(data_a, vocab[val1])
    # freq_norm_val2, count_vocab_val2, top_freq_val2 = extract_freqs(data_b, vocab[val2])

    NN_scores()
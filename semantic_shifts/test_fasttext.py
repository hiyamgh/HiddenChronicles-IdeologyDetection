import fasttext
import argparse
import operator
from collections import defaultdict
from tqdm import tqdm
import pickle
from nltk.util import ngrams
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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


def get_intersection_with_ocr_errors_ngram(neighs1, neighs2):
    common = set()
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
    len_scores_old = 0
    pbar = tqdm(total=len(vocab[val1]))
    for i, w in enumerate(vocab[val1]):
        # if w not in s_words and w in freq1 and w in freq2 and count1[w] > MIN_COUNT and count2[w] > MIN_COUNT:
        # if w in freq1 and w in freq2 and count1[w] > MIN_COUNT and count2[w] > MIN_COUNT:
            # neighbors_bef = set(topK(w, space1, args.k, count1, 100))
            # neighbors_aft = set(topK(w, sapce2, args.k, count2, 100))

        if w in vocab[val1] and w in vocab[val2]:
            neighbors_bef = set(out[1] for out in model1.get_nearest_neighbors(w, args.k))
            neighbors_aft = set(out[1] for out in model2.get_nearest_neighbors(w, args.k))
            # common = get_intersection_with_ocr_errors(neighs1=neighbors_bef, neighs2=neighbors_aft)
            common = get_intersection_with_ocr_errors_ngram(neighs1=neighbors_bef, neighs2=neighbors_aft)
            nn_scores.append((len(common), w))

        if len(nn_scores) >= len_scores_old + 100:
            with open('nn_scores_{}_{}_{}.pkl'.format(val1, val2, len(nn_scores)), 'wb') as f:
                pickle.dump(sorted(nn_scores), f)
            len_scores_old = len(nn_scores)

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
    parser.add_argument("--embed_a",
                        default='../Train_Word_Embedidng/fasttext/nahar/SGNS/'
                                'ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin',
                        help="prefix for embedding for split a")
    parser.add_argument("--embed_b",
                        default='../Train_Word_Embedidng/fasttext/assafir/SGNS/'
                                'ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin',
                        help="prefix for embedding for split b")
    parser.add_argument("--data_a",
                        default='../Train_Word_Embedidng/fasttext/data/nahar/2006.txt',
                        help="name of tokenized data file for split a")
    parser.add_argument("--data_b",
                        default='../Train_Word_Embedidng/fasttext/data/assafir/2006.txt',
                        help="name of tokenized data file for split b")
    parser.add_argument("--name_split_a",
                        default='nahar_2006',
                        help="short name for split a")
    parser.add_argument("--name_split_b",
                        default='assafir_2006',
                        help="short name for split b")
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

    model1 = fasttext.load_model(args.embed_a)
    model2 = fasttext.load_model(args.embed_b)

    MIN_COUNT = args.min_count

    val1, val2 = args.name_split_a, args.name_split_b

    vocab = {}
    vocab[val1] = model1.words
    vocab[val2] = model2.words

    NN_scores()
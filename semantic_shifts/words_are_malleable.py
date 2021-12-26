import fasttext
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
import argparse

# since we have fasttext embeddings, we don't need to get the
# words in the intersection of two embeddings, just calculate similarity
# for a given word as input, across several iterations t


def mkdir(folder):
    """ creates a directory if it doesn't already exist """
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_cosine_sim(v1, v2):
    """ Get the cosine similarity between two vectors """
    return dot(v1, v2) / (norm(v1) * norm(v2))


def get_stability_linear(model1, model2, subwords):

    def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
        '''
        Inputs:
            X: a matrix of dimension (m,n) where the columns are the English embeddings.
            Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
            train_steps: positive int - describes how many steps will gradient descent algorithm do.
            learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
        Outputs:
            R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
        '''
        # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
        # R is a square matrix with length equal to the number of dimensions in th  word embedding
        R = np.random.rand(X.shape[1], X.shape[1])

        for i in range(train_steps):
            if i % 25 == 0:
                print(f"loss at iteration {i} is: {compute_loss(X, Y, R):.4f}")
            # use the function that you defined to compute the gradient
            gradient = compute_gradient(X, Y, R)

            # update R by subtracting the learning rate times gradient
            R -= learning_rate * gradient
        return R

    def compute_gradient(X, Y, R):
        '''
        Inputs:
           X: a matrix of dimension (m,n) where the columns are the English embeddings.
           Y: a matrix of dimension (m,n) where the columns correspond to the French embeddings.
           R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
        Outputs:
           g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
        '''
        # m is the number of rows in X
        rows, columns = X.shape

        # gradient is X^T(XR - Y) * 2/m
        gradient = (np.dot(X.T, np.dot(X, R) - Y) * 2) / rows
        assert gradient.shape == (columns, columns)

        return gradient

    def compute_loss(X, Y, R):
        '''
        Inputs:
           X: a matrix of dimension (m,n) where the columns are the English embeddings.
           Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
           R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
        Outputs:
           L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
        '''
        # m is the number of rows in X
        m = len(X)

        # diff is XR - Y
        diff = np.dot(X, R) - Y

        # diff_squared is the element-wise square of the difference
        diff_squared = diff ** 2

        # sum_diff_squared is the sum of the squared elements
        sum_diff_squared = diff_squared.sum()

        # loss is the sum_diff_squared divided by the number of examples (m)
        loss = sum_diff_squared / m
        return loss

    # create the matrices X and Y of source embeddings i and target embeddings j
    X, Y = [], []
    for w in subwords:
        x = model1.get_word_vector(w) if ' ' not in w else model1.get_sentence_vector(w)
        y = model2.get_word_vector(w) if ' ' not in w else model2.get_sentence_vector(w)

        X.append(x)
        Y.append(y)

    X = np.vstack(X)
    Y = np.vstack(Y)

    # get the transformation matrix R
    R = align_embeddings(X=X, Y=Y)

    return R


def get_stability_combined(model1, model2, words_path=None, lmbda=0.5, k=50, t=5, save_dir='results/', file_name='stabilities_combined'):
    # get the intersection of the vocabularies of all models
    # note that this method can be applied to get the stability
    # of words from more than 2 models
    all_vocab = []
    all_vocab.append(model1.words)
    all_vocab.append(model2.words)

    # get the intersection of all vocabulary
    common_vocab = list(set.intersection(*map(set, all_vocab)))
    print('len of common vocab: {}'.format(len(common_vocab)))

    # initialize the 'graph', in this case it'll be a dictionary mapping a vocabulary
    # word to its stability values at range: 0: t-1
    stabilities = {}
    for w in common_vocab:
        stabilities[w] = np.ones_like(np.arange(t + 1, dtype=float))

    if words_path is not None:
        with open(words_path, 'r', encoding='utf-8') as f:
            words = f.readlines()

        words = [w[:-1] for w in words if '\n' in w]  # remove '\n'

        for w in words:
            if w not in stabilities:  # if its not in stabilities its not in common_words
                stabilities[w] = np.ones_like(np.arange(t + 1, dtype=float))
                common_vocab.append(w)

    for i in range(1, t + 1):
        for w in common_vocab:
            nnsims1 = model1.get_nearest_neighbors(w, k)
            nnsims2 = model2.get_nearest_neighbors(w, k)

            nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
            nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

            sim1, sim2 = 0, 0
            count_oov1, count_oov2 = 0, 0

            inter = set.intersection(*map(set, [nn1, nn2]))
            ranks1, ranks2 = [], []

            for wp in inter:
                ranks1.append(nn2.index(wp)) # index of wp in nn2
                ranks2.append(nn1.index(wp)) # index of wp in nn1

                if wp not in stabilities:
                    stabilities[wp] = np.ones_like(np.arange(t + 1, dtype=float))

            Count_neig12 = (len(nn1) * len(inter)) - sum([ranks1[z]/stabilities[w][i] for z in range(len(ranks1))])
            Count_neig21 = (len(nn2) * len(inter)) - sum([ranks2[z]/stabilities[w][i] for z in range(len(ranks2))])

            s_lin12 = None
            s_lin21 = None

            st_neig = (Count_neig12 + Count_neig21) / (2 * sum([i for i in range(1, k+1)])) # this is 2 * (n)(n+1)
            st_lin = np.mean([s_lin12, s_lin21])
            # final stability
            st = (lmbda * st_neig) + (lmbda * st_lin)
            stabilities[w][i] = st

            # min-max normalize st to fall in the 0-1 range
            # do so by min-max normalizing st taking values 0:i
            stabilities[w] = (stabilities[w] - stabilities[w].min()) / (stabilities[w].max() - stabilities[w].min())

            print('w: {}'.format(w))
            print('sim 1: {}, sim2: {}, st: {}, normalized st: {}'.format(sim1, sim2, st, stabilities[w][i]))
            print('oov1/nn1={}'.format(count_oov1 / len(nn1)))
            print('oov2/nn2={}'.format(count_oov2 / len(nn2)))
            print('===============================================================')

    # save the stabilities dictionary for loading it later on
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


def get_stability_neighbors(model1, model2, words_path=None, k=50, t=5, save_dir='results/', file_name='stabilities_neighbors'):
    """
    Algorithm 1 in 'Words are malleable' paper, which states that
    the similarity of two words is known by the similarities of their
    neighbours. The aim of this is to see to which extent the neighbors
    of the word in different embeddings are similar

    :param model1: first embedding corpus
    :param model2: second embedding corpus
    :param k: the number of top neighbours to consider
    :param t: the number of iterations
    :param save_dir: directory to save the stabilities dictionary in
    :param file_name: name of the file to save
    :return:
    """
    # no need for getting common vocab as the neighbors of an OOV
    # words are also OOV, so we won't find them in the common
    # vocabulary. I don't think we can also loop over all words in the
    # common vocabulary because we have no common vocabulary

    # get the intersection of the vocabularies of both models
    all_vocab = []
    all_vocab.append(model1.words)
    all_vocab.append(model2.words)
    # get the intersection of all vocabulary
    common_vocab = list(set.intersection(*map(set, all_vocab)))
    print('len of common vocab: {}'.format(len(common_vocab)))

    # initialize the 'graph', in this case it'll be a dictionary mapping a vocabulary
    # word to its stability values at range: 0: t-1
    stabilities = {}
    for w in common_vocab:
        stabilities[w] = np.ones_like(np.arange(t + 1, dtype=float))

    # if there are some set of words that we are interested in
    # and these words are OOV, we can also get their neighbors
    # (Although I still have to see the issue raised on github: 'Neighbors of OOV are also OOV')
    if words_path is not None:
        with open(words_path, 'r', encoding='utf-8') as f:
            words = f.readlines()

        words = [w[:-1] for w in words if '\n' in w] # remove '\n'

        for w in words:
            if w not in stabilities: # if its not in stabilities its not in common_words
                stabilities[w] = np.ones_like(np.arange(t + 1, dtype=float))
                common_vocab.append(w)

    for i in range(1, t+1):
        for w in common_vocab:
            nnsims1 = model1.get_nearest_neighbors(w, k)
            nnsims2 = model2.get_nearest_neighbors(w, k)

            nn1 = [n[1] for n in nnsims1] # get only the neighbour, not the similarity
            nn2 = [n[1] for n in nnsims2] # get only the neighbour, not the similarity

            sim1, sim2 = 0, 0
            count_oov1, count_oov2 = 0, 0
            for wp in nn2:
                # wp = re.sub('\n', '', wp)
                w_v = model1.get_word_vector(w) if ' ' not in w else model1.get_sentence_vector(w)
                wp_v = model1.get_sentence_vector(wp) if ' ' not in wp else model1.get_sentence_vector(wp)
                if wp not in stabilities:
                    stabilities[wp] = np.ones_like(np.arange(t+1, dtype=float))
                    count_oov2 += 1
                sim2 += get_cosine_sim(w_v, wp_v) * stabilities[wp][i-1]
            sim2 /= len(nn2)

            for wp in nn1:
                # wp = re.sub('\n', '', wp)
                w_v = model2.get_word_vector(w) if ' ' not in w else model2.get_sentence_vector(w)
                wp_v = model2.get_sentence_vector(wp) if ' ' not in wp else model2.get_sentence_vector(wp)
                if wp not in stabilities:
                    stabilities[wp] = np.ones_like(np.arange(t+1, dtype=float))
                    count_oov1 += 1
                sim1 += get_cosine_sim(w_v, wp_v) * stabilities[wp][i-1]
            sim1 /= len(nn1)

            # calculate stability as the average of both similarities
            st = np.mean([sim1, sim2])
            stabilities[w][i] = st

            # min-max normalize st to fall in the 0-1 range
            # do so by min-max normalizing st taking values 0:i
            stabilities[w] = (stabilities[w] - stabilities[w].min()) / (stabilities[w].max() - stabilities[w].min())

            print('w: {}'.format(w))
            print('sim 1: {}, sim2: {}, st: {}, normalized st: {}'.format(sim1, sim2, st, stabilities[w][i]))
            print('oov1/nn1={}'.format(count_oov1 / len(nn1)))
            print('oov2/nn2={}'.format(count_oov2 / len(nn2)))
            print('===============================================================')

    # save the stabilities dictionary for loading it later on
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', default='E:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/', help='path to trained models files for first embedding')
    parser.add_argument('--path2', default='E:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/', help='path to trained models files for secnd embedding')
    parser.add_argument("--model1", default='1975.bin', help="model 1 name")
    parser.add_argument("--model2", default='2007.bin', help="model 2 name")
    parser.add_argument("--words_file", default="input/keywords.txt", help="list of words interested in getting their stability values")
    parser.add_argument("--k", default=100, help="number of nearest neighbors to consider per word")
    parser.add_argument("--t", default=5, help="number of iterations to consider for the neighbours algorithm")
    parser.add_argument("--save_dir", default="results/", help="directory to save stabilities dictionary")
    parser.add_argument("--file_name", default="stabilities", help="file name of the stabilities dictionary to be saved")
    args = parser.parse_args()

    model1 = fasttext.load_model(os.path.join(args.path1, args.model1))
    model2 = fasttext.load_model(os.path.join(args.path2, args.model2))
    keywords_path = args.words_file
    k = args.k
    t = args.t
    save_dir = args.save_dir
    file_name = args.file_name

    stabilities = get_stability_neighbors(model1, model2, words_path=keywords_path, k=k, t=t, save_dir=save_dir, file_name=file_name)


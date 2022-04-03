import fasttext
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
import argparse
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


def mkdir(folder):
    """ creates a directory if it doesn't already exist """
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_cosine_sim(v1, v2):
    """ Get the cosine similarity between two vectors """
    return dot(v1, v2) / (norm(v1) * norm(v2))


def load_linear_mapping_matrices(dir_name, mat_name, model1_name, model2_name):
    """
    load the linear transformation matrices between one embedding and another
    :param dir_name: name of the directory where the matrices are stored
    :param mat_name: file name used to store the matrices (this is a suffix; the
                    inverse transformation matrix has an extra substring '_inv')
    :return the transformation matrix R and its inverse R_inv
    """
    R = np.load(os.path.join(dir_name, mat_name + '_{}_{}.npy').format(model1_name, model2_name))
    R_inv = np.load(os.path.join(dir_name, mat_name + '_{}_{}_inv.npy'.format(model1_name, model2_name)))
    return R, R_inv


def get_stability_linear_mapping_one_word(model1, model2, model1_name, model2_name, mat_name, dir_name_matrices, w):
    ''' gets the stability values of one oword  only
        assumes that transformation matrices are already computed
    '''
    R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=model1_name, model2_name=model2_name)

    w0 = model1.get_word_vector(w) if ' ' not in w else model1.get_sentence_vector(w)
    w1 = model2.get_word_vector(w) if ' ' not in w else model2.get_sentence_vector(w)

    # the stability of a word is basically the stability of the vector to its mapped
    # vector after applying the mapping back and forth
    sim01 = get_cosine_sim(R_inv.dot(R.dot(w0)), w0)
    sim10 = get_cosine_sim(R.dot(R_inv.dot(w1)), w1)

    stability = (sim01 + sim10) / 2

    print('stability (linear) of the word {}: {}'.format(w, stability))

    return sim01, sim10


def learn_stability_matrices(model1, model2, model1_name, model2_name, subwords, num_steps, mat_name):
    """
    Get the stability by applying a transformation matrix that maps word w in embedding space 1
    to word w in embedding space 2. Transformation matrix is learned through gradient descent optimization
    by applying Forbenious norm in order to minimize loss. It uses a set of training words (stopwords or
    any set of frequent words that have a stable meaning regardless of time/source)

    :param model1: trained model - embedding 1
    :param model2: trained model - embedding 2
    :param subwords: list of words/stopwords to consider for learning the transformation matrix
    :param num_steps: number of training steps for gradient descent optimization
    :param w: the word we want to get the stability for
    :return:
    """
    # print('Getting linear stability for the word {}'.format(w))
    print('Calculating stability values (linear)')

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

        losses = []
        t1 = time.time()
        for i in range(train_steps):
            loss = compute_loss(X, Y, R)

            if i % 25 == 0:
                print(f"loss at iteration {i} is: {loss:.4f}")

            losses.append(loss)
            # use the function that you defined to compute the gradient
            gradient = compute_gradient(X, Y, R)

            # update R by subtracting the learning rate times gradient
            R -= learning_rate * gradient
        t2 = time.time()
        print('time taken to learn the transformation matrix: {} mins'.format((t2-t1)/60))
        return R, losses

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

    def get_transformation_matrices():
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
        R, losses = align_embeddings(X=X, Y=Y, train_steps=num_steps)
        R_inv = np.linalg.inv(R)

        return R, R_inv, losses

    def save_matrices(R, R_inv):
        # save transformation matrix and its inverse
        np.save(os.path.join(dir_name_matrices, mat_name + '_{}_{}.npy'.format(model1_name, model2_name)), R)
        np.save(os.path.join(dir_name_matrices, mat_name + '_{}_{}_inv.npy'.format(model1_name, model2_name)), R_inv)

    def plot_losses(losses):
        # dir_name_losses = os.path.join(save_dir, 'loss_plots/')
        # mkdir(losses)  # create directory of does not already exist
        plt.plot(range(len(losses)), losses)
        plt.xlabel('steps')
        plt.ylabel('Loss')

        plt.savefig(os.path.join(dir_name_losses, mat_name + '_{}_{}.png'.format(model1_name, model2_name)))
        plt.close()

    # if the matrices exist, return since they will be loaded in the combined method
    if os.path.exists(os.path.join(dir_name_matrices, mat_name + '_{}_{}.npy'.format(model1_name, model2_name))):
        print('matrices found in {}'.format(os.path.join(dir_name_matrices, mat_name + '_{}_{}.npy'.format(model1_name, model2_name))))
        return

    # get the transformation matrix and its inverse
    R, R_inv, losses = get_transformation_matrices()
    print('got transformation matrices ...')

    plot_losses(losses)
    print('saved plot of loss ...')

    # save the transformation matrices
    save_matrices(R=R, R_inv=R_inv)
    print('saved transformation matrices ...')
    return


def get_stability_linear_mapping(model1, model2, model1_name, model2_name, mat_name, words_path=None, save_dir='results/', file_name='stabilities_linear'):
    print('Calculating stability values (linear)')

    if words_path is None:
        all_vocab = []
        all_vocab.append(model1.words)
        all_vocab.append(model2.words)
        vocab = list(set.intersection(*map(set, all_vocab)))

    else:
        with open(words_path, 'r', encoding='utf-8') as f:
            words = f.readlines()
        words = [w[:-1] for w in words if '\n' in w]  # remove '\n'
        words = [w for w in words if w.strip() != '']
        vocab = [w for w in words]

    print('len of vocab: {}'.format(len(vocab)))
    # dictionary mapping each word w to its stability value. Before any processing, stability of any word w is 1.0
    stabilities = {}
    for w in vocab:
        stabilities[w] = 1.0

    for w in vocab:
        s_lin12, s_lin21 = get_stability_linear_mapping_one_word(model1, model2, model1_name=model1_name, model2_name=model2_name,
                                                                 mat_name=mat_name, dir_name_matrices=dir_name_matrices, w=w)

        stability = (s_lin12 + s_lin21) / 2
        stabilities[w] = stability

    # save the stabilities dictionary
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


def get_stability_combined_one_word(w, models, models_names, mat_name, dir_name_matrices, k=50):
    """ gets the combined stability value of a certain word w """
    if len(models) > 2:
        similarities = []
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    nnsims1 = models[i].get_nearest_neighbors(w, k)
                    nnsims2 = models[j].get_nearest_neighbors(w, k)

                    nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
                    nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

                    inter = set.intersection(*map(set, [nn1, nn2]))

                    ranks1, ranks2 = {}, {}  # for storing indices
                    not_found1, not_found2 = [], []  # for storing words that are found in one list, but not in the other

                    # loop over neighbors of w in embedding space 1, check if they're in the neighbors of w in embedding space 2
                    # calculate their rank, if yes.
                    for wp in nn1:
                        if wp in nn2:
                            ranks1[wp] = nn2.index(wp)  # index of wp in nn2
                        else:
                            # if not present, it has no index
                            not_found1.append(wp)

                    # loop over neighbors of w in embedding space 2, check if they're in the neighbors of w in embedding space 1
                    # calculate their rank, if yes.
                    for wp in nn2:
                        if wp in nn1:
                            ranks2[wp] = nn1.index(wp)  # index of wp in nn1
                        else:
                            # if not present, it has no index
                            not_found2.append(wp)

                    sum_ranks1, sum_ranks2 = 0.0, 0.0
                    for wp in ranks1:
                        sum_ranks1 += ranks1[wp]
                    for wp in ranks2:
                        sum_ranks2 += ranks2[wp]

                    Count_neig12 = (len(nn1) * len(inter)) - sum_ranks1
                    Count_neig21 = (len(nn2) * len(inter)) - sum_ranks2

                    # if there are some words that are found in one list but not in the other
                    # then calculate their stability using linear mapping approach
                    if not_found1 != [] or not_found2 != []:
                        R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=models_names[i], model2_name=models_names[j])

                        sim_lin01, sim_lin10 = 0.0, 0.0
                        for wp in not_found1:
                            w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                            wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                            val = get_cosine_sim(R.dot(wp_v), w_v)
                            sim_lin01 += val
                            print('{} - {} - {}'.format(w, wp, val))
                        sim_lin01 /= len(not_found1)

                        for wp in not_found2:
                            w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                            wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                            val = get_cosine_sim(R_inv.dot(wp_v), w_v)
                            sim_lin10 += val
                            print('{} - {} - {}'.format(w, wp, val))
                        sim_lin10 /= len(not_found2)

                    st_neig = (Count_neig12 + Count_neig21) / (2 * sum(
                        [i for i in range(1, k + 1)]))  # this is 2 * (k)(k+1) where k is the number of nearest neighbors
                    st_lin = None
                    if not_found1 != [] or not_found2 != []:
                        st_lin = np.mean([sim_lin01, sim_lin10])

                    # calculate value of lambda
                    if nn1 == nn2:
                        # when the nearest neighbours of w are exactly the same,
                        # and have the same order in embedding 1 and embedding 2
                        # lmbda = 1.0
                        st = st_neig
                        print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, st_neig, '-', st))
                    elif Count_neig12 == 0 and Count_neig12 == 0:
                        # when the nearest neighbours in embedding 1 are completely
                        # not found in embedding 2, and vice versa would be also true
                        # lmbda = 0
                        st = st_lin
                        print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, '-', st_lin, st))
                    else:
                        # some neighbours of w in embedding 1 are found in embedding 2,
                        # and vice versa would be true
                        lmbda = 0.5
                        st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
                        print('{}: st_nei: {}, st_lin: {}, st: {}'.format(w, st_neig, st_lin, st))

                    similarities.append(st)

        return np.mean(similarities)

    else:
        nnsims1 = models[0].get_nearest_neighbors(w, k)
        nnsims2 = models[1].get_nearest_neighbors(w, k)

        nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
        nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

        inter = set.intersection(*map(set, [nn1, nn2]))

        ranks1, ranks2 = {}, {}  # for storing indices
        not_found1, not_found2 = [], []  # for storing words that are found in one list, but not in the other

        # loop over neighbors of w in embedding space 1, check if they're in the neighbors of w in embedding space 2
        # calculate their rank, if yes.
        for wp in nn1:
            if wp in nn2:
                ranks1[wp] = nn2.index(wp)  # index of wp in nn2

            else:
                # if not present, it has no index
                not_found1.append(wp)

        # loop over neighbors of w in embedding space 2, check if they're in the neighbors of w in embedding space 1
        # calculate their rank, if yes.
        for wp in nn2:
            if wp in nn1:
                ranks2[wp] = nn1.index(wp)  # index of wp in nn1

            else:
                # if not present, it has no index
                not_found2.append(wp)

        sum_ranks1, sum_ranks2 = 0.0, 0.0
        for wp in ranks1:
            sum_ranks1 += ranks1[wp]
        for wp in ranks2:
            sum_ranks2 += ranks2[wp]

        Count_neig12 = (len(nn1) * len(inter)) - sum_ranks1
        Count_neig21 = (len(nn2) * len(inter)) - sum_ranks2

        # if there are some words that are found in one list but not in the other
        # then calculate their stability using linear mapping approach
        if not_found1 != [] or not_found2 != []:
            R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name,
                                                    model1_name=models_names[0], model2_name=models_names[1])

            sim_lin01, sim_lin10 = 0.0, 0.0
            for wp in not_found1:
                w_v = models[1].get_word_vector(w) if ' ' not in w else models[1].get_sentence_vector(w)
                wp_v = models[0].get_word_vector(wp) if ' ' not in wp else models[0].get_sentence_vector(wp)

                val = get_cosine_sim(R.dot(wp_v), w_v)
                sim_lin01 += val
                # print('{} - {} - {}'.format(w, wp, val))
            sim_lin01 /= len(not_found1)

            for wp in not_found2:
                w_v = models[0].get_word_vector(w) if ' ' not in w else models[0].get_sentence_vector(w)
                wp_v = models[1].get_word_vector(wp) if ' ' not in wp else models[1].get_sentence_vector(wp)

                val = get_cosine_sim(R_inv.dot(wp_v), w_v)
                sim_lin10 += val
                # print('{} - {} - {}'.format(w, wp, val))
            sim_lin10 /= len(not_found2)

        st_neig = (Count_neig12 + Count_neig21) / (2 * sum(
            [i for i in range(1, k + 1)]))  # this is 2 * (k)(k+1) where k is the number of nearest neighbors
        st_lin = None
        if not_found1 != [] or not_found2 != []:
            st_lin = np.mean([sim_lin01, sim_lin10])

        # calculate value of lambda
        if nn1 == nn2:
            # when the nearest neighbours of w are exactly the same,
            # and have the same order in embedding 1 and embedding 2
            # lmbda = 1.0
            st = st_neig
            print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, st_neig, '-', st))
        elif Count_neig12 == 0 and Count_neig12 == 0:
            # when the nearest neighbours in embedding 1 are completely
            # not found in embedding 2, and vice versa would be also true
            # lmbda = 0
            st = st_lin
            print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, '-', st_lin, st))
        else:
            # some neighbours of w in embedding 1 are found in embedding 2,
            # and vice versa would be true
            lmbda = 0.5
            st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
            print('{}: st_nei: {}, st_lin: {}, st: {}'.format(w, st_neig, st_lin, st))

        return st


def get_stability_combined(models, models_names, mat_name, words_path=None, k=50, save_dir='results/',
                           file_name='stabilities_combined'):
    """
    Method that applies Algorithm 2 (Combined method) of Words are Malleable paper by Azarbonyad et al (2017)
    However, we have modified the code from the one presented in the paper in few places
    because of the fact that we can get vectors of words that are OOV (out-of-vocabulary)
    therefore, no need for getting the words in the intersection of two embedding spaces as
    most of the words we were interested in are in fact OOV due to spelling errors
    caused by applying OCR to digitized historical documents.

    Also, the code is formatted to work with t=1, i.e., we are only using direct neighbors of words (t=1)
    and not neighbors of neighbors (t >= 1). Therefore, we removed MinMax normalization step

    :param model1: fasttext model representing embedding space 1
    :param model2: fasttext model representing embedding space 2
    :param mat_name: name of the transformation matrix calculated via linear mapping that maps from embedding space 1 to embedding space 2
    :param words_path: path to file that contains words that are OOV
    :param k: number of nearest neighbors to extract per word
    :param t: number of iterations
    :param save_dir: directory to save final stabilities per word (saved as a python dictionary)
    :param file_name: name of the final stabilities file (saved as a python dictionary)
    :return:
    """
    print('Calculating stability values (combined) - k={}'.format(k))
    if words_path is None:
        all_vocab = []
        all_vocab.append(model1.words)
        all_vocab.append(model2.words)
        vocab = list(set.intersection(*map(set, all_vocab)))

    else:
        with open(words_path, 'r', encoding='utf-8') as f:
            words = f.readlines()
        words = [w[:-1] for w in words if '\n' in w]  # remove '\n'
        words = [w for w in words if w.strip() != '']
        vocab = [w.strip() for w in words]

    print('len of vocab: {}'.format(len(vocab)))
    stabilities = {}
    for w in vocab:
        if len(models) > 2:
            similarities = []
            for i in range(len(models)):
                for j in range(len(models)):
                    if i != j:
                        nnsims1 = models[i].get_nearest_neighbors(w, k)
                        nnsims2 = models[j].get_nearest_neighbors(w, k)

                        nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
                        nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

                        inter = set.intersection(*map(set, [nn1, nn2]))

                        ranks1, ranks2 = {}, {}  # for storing indices
                        not_found1, not_found2 = [], []  # for storing words that are found in one list, but not in the other

                        # loop over neighbors of w in embedding space 1, check if they're in the neighbors of w in embedding space 2
                        # calculate their rank, if yes.
                        for wp in nn1:
                            if wp in nn2:
                                ranks1[wp] = nn2.index(wp)  # index of wp in nn2
                            else:
                                # if not present, it has no index
                                not_found1.append(wp)

                        # loop over neighbors of w in embedding space 2, check if they're in the neighbors of w in embedding space 1
                        # calculate their rank, if yes.
                        for wp in nn2:
                            if wp in nn1:
                                ranks2[wp] = nn1.index(wp)  # index of wp in nn1
                            else:
                                # if not present, it has no index
                                not_found2.append(wp)

                        sum_ranks1, sum_ranks2 = 0.0, 0.0
                        for wp in ranks1:
                            sum_ranks1 += ranks1[wp]
                        for wp in ranks2:
                            sum_ranks2 += ranks2[wp]

                        Count_neig12 = (len(nn1) * len(inter)) - sum_ranks1
                        Count_neig21 = (len(nn2) * len(inter)) - sum_ranks2

                        # if there are some words that are found in one list but not in the other
                        # then calculate their stability using linear mapping approach
                        if not_found1 != [] or not_found2 != []:
                            R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=models_names[i], model2_name=models_names[j])

                            sim_lin01, sim_lin10 = 0.0, 0.0
                            for wp in not_found1:
                                w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                                wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                                val = get_cosine_sim(R.dot(wp_v), w_v)
                                sim_lin01 += val
                                # print('{} - {} - {}'.format(w, wp, val))
                            sim_lin01 /= len(not_found1)

                            for wp in not_found2:
                                w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                                wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                                val = get_cosine_sim(R_inv.dot(wp_v), w_v)
                                sim_lin10 += val
                                # print('{} - {} - {}'.format(w, wp, val))
                            sim_lin10 /= len(not_found2)

                        st_neig = (Count_neig12 + Count_neig21) / (2 * sum([i for i in range(1,
                                                                                             k + 1)]))  # this is 2 * (k)(k+1) where k is the number of nearest neighbors
                        st_lin = None
                        if not_found1 != [] or not_found2 != []:
                            st_lin = np.mean([sim_lin01, sim_lin10])

                        # calculate value of lambda
                        if nn1 == nn2:
                            # when the nearest neighbours of w are exactly the same,
                            # and have the same order in embedding 1 and embedding 2
                            # lmbda = 1.0
                            st = st_neig
                            print('{}-{}: {}: st_neigh: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], w, st_neig, '-', st))
                        elif Count_neig12 == 0 and Count_neig12 == 0:
                            # when the nearest neighbours in embedding 1 are completely
                            # not found in embedding 2, and vice versa would be also true
                            # lmbda = 0
                            st = st_lin
                            print('{}-{}: {}: st_neigh: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], w, '-', st_lin, st))
                        else:
                            # some neighbours of w in embedding 1 are found in embedding 2,
                            # and vice versa would be true
                            lmbda = 0.5
                            st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
                            print('{}-{}: {}: st_nei: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], w, st_neig, st_lin, st))

                        similarities.append(st)

            stabilities[w] = np.mean(similarities)
            print('final combined stability of: {} = {}'.format(w, np.mean(similarities)))

        else:
            nnsims1 = models[0].get_nearest_neighbors(w, k)
            nnsims2 = models[1].get_nearest_neighbors(w, k)

            nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
            nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

            inter = set.intersection(*map(set, [nn1, nn2]))

            ranks1, ranks2 = {}, {}  # for storing indices
            not_found1, not_found2 = [], []  # for storing words that are found in one list, but not in the other

            # loop over neighbors of w in embedding space 1, check if they're in the neighbors of w in embedding space 2
            # calculate their rank, if yes.
            for wp in nn1:
                if wp in nn2:
                    ranks1[wp] = nn2.index(wp)  # index of wp in nn2
                else:
                    # if not present, it has no index
                    not_found1.append(wp)

            # loop over neighbors of w in embedding space 2, check if they're in the neighbors of w in embedding space 1
            # calculate their rank, if yes.
            for wp in nn2:
                if wp in nn1:
                    ranks2[wp] = nn1.index(wp)  # index of wp in nn1
                else:
                    # if not present, it has no index
                    not_found2.append(wp)

            sum_ranks1, sum_ranks2 = 0.0, 0.0
            for wp in ranks1:
                sum_ranks1 += ranks1[wp]
            for wp in ranks2:
                sum_ranks2 += ranks2[wp]

            Count_neig12 = (len(nn1) * len(inter)) - sum_ranks1
            Count_neig21 = (len(nn2) * len(inter)) - sum_ranks2

            # if there are some words that are found in one list but not in the other
            # then calculate their stability using linear mapping approach
            if not_found1 != [] or not_found2 != []:
                R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=models_names[0], model2_name=models_names[1])

                sim_lin01, sim_lin10 = 0.0, 0.0
                for wp in not_found1:
                    w_v = models[1].get_word_vector(w) if ' ' not in w else models[1].get_sentence_vector(w)
                    wp_v = models[0].get_word_vector(wp) if ' ' not in wp else models[0].get_sentence_vector(wp)

                    val = get_cosine_sim(R.dot(wp_v), w_v)
                    sim_lin01 += val
                    # print('{} - {} - {}'.format(w, wp, val))
                sim_lin01 /= len(not_found1)

                for wp in not_found2:
                    w_v = models[0].get_word_vector(w) if ' ' not in w else models[0].get_sentence_vector(w)
                    wp_v = models[1].get_word_vector(wp) if ' ' not in wp else models[1].get_sentence_vector(wp)

                    val = get_cosine_sim(R_inv.dot(wp_v), w_v)
                    sim_lin10 += val
                    # print('{} - {} - {}'.format(w, wp, val))
                sim_lin10 /= len(not_found2)

            st_neig = (Count_neig12 + Count_neig21) / (2 * sum([i for i in range(1, k + 1)]))  # this is 2 * (k)(k+1) where k is the number of nearest neighbors
            st_lin = None
            if not_found1 != [] or not_found2 != []:
                st_lin = np.mean([sim_lin01, sim_lin10])

            # calculate value of lambda
            if nn1 == nn2:
                # when the nearest neighbours of w are exactly the same,
                # and have the same order in embedding 1 and embedding 2
                # lmbda = 1.0
                st = st_neig
                print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, st_neig, '-', st))
            elif Count_neig12 == 0 and Count_neig12 == 0:
                # when the nearest neighbours in embedding 1 are completely
                # not found in embedding 2, and vice versa would be also true
                # lmbda = 0
                st = st_lin
                print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, '-', st_lin, st))
            else:
                # some neighbours of w in embedding 1 are found in embedding 2,
                # and vice versa would be true
                lmbda = 0.5
                st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
                print('{}: st_nei: {}, st_lin: {}, st: {}'.format(w, st_neig, st_lin, st))

            stabilities[w] = st
            print('final combined stability of: {} = {}'.format(w, st))

    # save the stabilities dictionary
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


def get_stability_neighbors_one_word(w, models, k=50):
    similarities = []
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                nnsims1 = models[i].get_nearest_neighbors(w, k)
                nnsims2 = models[j].get_nearest_neighbors(w, k)

                nn1 = [n[1] for n in nnsims1]  # get only the neighbor, not the similarity
                nn2 = [n[1] for n in nnsims2]  # get only the neighbor, not the similarity

                sim1, sim2 = 0, 0
                for wp in nn2:
                    w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                    wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                    sim2 += get_cosine_sim(w_v, wp_v)

                sim2 /= len(nn2)

                for wp in nn1:
                    w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                    wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                    sim1 += get_cosine_sim(w_v, wp_v)

                sim1 /= len(nn1)

                # calculate stability as the average of both similarities
                similarities.append(sim1)
                similarities.append(sim2)

    # calculate the word's stability
    st = np.mean(similarities)
    print('Neighbor\'s approach stability: {}: {}'.format(w, st))


def get_stability_neighbors(models, words_path=None, k=50, save_dir='results/', file_name='stabilities_neighbors'):
    """
    Method that applies Algorithm 1 (Neighbors method) of Words are Malleable paper by Azarbonyad et al (2017)
    However, we have modified the code from the one presented in the paper in few places
    because of the fact that we can get vectors of words that are OOV (out-of-vocabulary),
    therefore, no need for getting the words in the intersection of two embedding spaces as
    most of the words we were interested in are in fact OOV due to spelling errors
    caused by applying OCR to digitized historical documents.

    Also, the code is formatted to work with t=1, i.e., we are only using direct neighbors of words (t=1)
    and not neighbors of neighbors (t >= 1). Therefore, we removed MinMax normalization step

    :param model1: fasttext model representing embedding space 1
    :param model2: fasttext model representing embedding space 1
    :param k: number of nearest neighbors to extract per word
    :param t: number of iterations
    :param save_dir: directory to save final stabilities per word (saved as a python dictionary)
    :param file_name: name of the final stabilities file (saved as a python dictionary)
    :return:
    """

    print('Calculating stability values (neighbors) - k={}'.format(k))
    if words_path is None:
        all_vocab = []
        all_vocab.append(model1.words)
        all_vocab.append(model2.words)
        vocab = list(set.intersection(*map(set, all_vocab)))
        print('len of vocab: {}'.format(len(vocab)))
    else:
        with open(words_path, 'r', encoding='utf-8') as f:
            words = f.readlines()
        words = [w[:-1] for w in words if '\n' in w] # remove '\n'
        words = [w for w in words if w.strip() != '']
        vocab = [w for w in words]
        print('len of vocab: {}'.format(len(vocab)))

    # dictionary mapping each word w to its stability value. Before any processing, stability of any word w is 1.0
    stabilities = {}

    for w in vocab:
        similarities = []
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    nnsims1 = models[i].get_nearest_neighbors(w, k)
                    nnsims2 = models[j].get_nearest_neighbors(w, k)

                    nn1 = [n[1] for n in nnsims1]  # get only the neighbor, not the similarity
                    nn2 = [n[1] for n in nnsims2]  # get only the neighbor, not the similarity

                    sim1, sim2 = 0, 0
                    for wp in nn2:
                        w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                        wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                        sim2 += get_cosine_sim(w_v, wp_v)

                    sim2 /= len(nn2)

                    for wp in nn1:
                        w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                        wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                        sim1 += get_cosine_sim(w_v, wp_v)

                    sim1 /= len(nn1)

                    # calculate stability as the average of both similarities
                    similarities.append(sim1)
                    similarities.append(sim2)

        # calculate the word's stability
        st = np.mean(similarities)
        stabilities[w] = st
        print('Neighbor\'s approach stability: {}: {}'.format(w, st))

    # save the stabilities dictionary
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


if __name__ == '__main__':
    print('hello')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', default='D:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/', help='path to trained models files for first embedding')
    parser.add_argument('--path2', default='D:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/', help='path to trained models files for second embedding')
    parser.add_argument('--path3', default=None, help='path to trained models files for second embedding')

    parser.add_argument("--model1", default='2007.bin', help="model 1 file name")
    parser.add_argument("--model2", default='2008.bin', help="model 2 file name")
    parser.add_argument("--model3", default=None, help="model 3 file name")

    parser.add_argument("--model1_name", default="nahar_07", help="string to name model 1 - used for saving results")
    parser.add_argument("--model2_name", default="nahar_08", help="string to name model 2 - used for saving results")
    parser.add_argument("--model3_name", default=None, help="string to name model 3 - used for saving results")

    # neighbor and combined approach (words_file is needed by linear approach as well)
    parser.add_argument("--words_file", default="from_DrFatima/sentiment_keywords.txt", help="list of words interested in getting their stability values")
    parser.add_argument("--k", default=100, help="number of nearest neighbors to consider per word - for neighbours and combined approach")
    parser.add_argument("--save_dir", default="results_diachronic_new/", help="directory to save stabilities dictionary")

    # linear approach
    parser.add_argument("--num_steps", default=80000, help="number of training steps for gradient descent optimization")
    parser.add_argument("--mat_name", default="trans", help="prefix of the name of the transformation matrices to be saved - linear mapping approach")

    # name of the method to be used ('combined' vs. 'linear' vs. neighbor)
    parser.add_argument("--method", default="combined", help="method to calculate stability - either combined/neighbors/linear")
    args = parser.parse_args()

    t1 = time.time()
    model1 = fasttext.load_model(os.path.join(args.path1, args.model1))
    print('loaded model 1: {}'.format(args.model1_name))
    model2 = fasttext.load_model(os.path.join(args.path2, args.model2))
    print('loaded model 2: {}'.format(args.model2_name))

    if args.path3 is not None:
        model3 = fasttext.load_model(os.path.join(args.path3, args.model3))
        print('loaded model 3: {}'.format(args.model3_name))
        models = [model1, model2, model3]
        models_names = [args.model1_name, args.model2_name, args.model3_name]
    else:
        models = [model1, model2]
        models_names = [args.model1_name, args.model2_name]

    t2 = time.time()
    print('time taken to load models: {}'.format((t2-t1)/60))

    model1_name = args.model1_name
    model2_name = args.model2_name

    if args.path3 is not None:
        model3_name = args.model3_name

    # neighbors approach
    keywords_path = args.words_file
    k = int(args.k)
    save_dir = args.save_dir

    stopwords_list = stopwords.words('arabic') # stopwords for linear mapping approach
    num_steps = args.num_steps # number of training steps for gradient descent
    mat_name = args.mat_name  # matrix name of the transformation matrix

    # create saving directories for combined, neighbors, and linear mapping approaches
    if args.path3 is None:
        save_dir_combined_neighbor = os.path.join(save_dir, '{}_{}/k{}/'.format(model1_name, model2_name, k)) # to save stability dictionaries of combined and neighbors approach
        save_dir_linear = os.path.join(save_dir, "{}_{}/linear_numsteps{}/".format(model1_name, model2_name, num_steps)) # to save transformation matrices
        save_dir_linear_stabilities = os.path.join(save_dir, '{}_{}/'.format(model1_name, model2_name)) # to save stability dictionary of linear mapping approach
    else:
        save_dir_combined_neighbor = os.path.join(save_dir, '{}_{}_{}/k{}/'.format(model1_name, model2_name, model3_name, k))  # to save stability dictionaries of combined and neighbors approach
        save_dir_linear = os.path.join(save_dir, "{}_{}_{}/linear_numsteps{}/".format(model1_name, model2_name, model3_name, num_steps))  # to save transformation matrices
        save_dir_linear_stabilities = os.path.join(save_dir, '{}_{}_{}/'.format(model1_name, model2_name, model3_name))  # to save stability dictionary of linear mapping approach

    # sub-directories for saving the transformation matrices
    dir_name_matrices = os.path.join(save_dir_linear, 'matrices/')
    dir_name_losses = os.path.join(save_dir_linear, 'loss_plots/')

    mkdir(save_dir_linear)             # create directory if does not already exist
    mkdir(save_dir_combined_neighbor)  # create directory if does not already exist
    mkdir(dir_name_matrices)           # create directory if does not already exist
    mkdir(dir_name_losses)             # create directory if does not already exist

    method = args.method

    if method == 'linear' or method == 'combined':
        if len(models) > 2:
            for i in range(len(models)):
                for j in range(len(models)):
                    if i != j:
                        # we need the transformation matrices for the combined approach because it uses linear mapping
                        learn_stability_matrices(model1=models[i], model2=models[j], model1_name=models_names[i], model2_name=models_names[j], subwords=stopwords_list,
                                                 num_steps=num_steps, mat_name=mat_name)

                        get_stability_linear_mapping(model1=models[i], model2=models[j], model1_name=models_names[i], model2_name=models_names[j], mat_name=mat_name,
                                                     words_path=keywords_path, save_dir=save_dir_linear_stabilities, file_name='stabilities_linear')
        else:
            learn_stability_matrices(model1=models[0], model2=models[1], model1_name=models_names[0],
                                     model2_name=models_names[1], subwords=stopwords_list,
                                     num_steps=num_steps, mat_name=mat_name)

            get_stability_linear_mapping(model1=models[0], model2=models[1], model1_name=models_names[0],
                                         model2_name=models_names[1], mat_name=mat_name,
                                         words_path=keywords_path, save_dir=save_dir_linear_stabilities,
                                         file_name='stabilities_linear')

    if method == "combined":
        # run the combined algorithm
        get_stability_combined(models=models, models_names=models_names, mat_name=mat_name, words_path=keywords_path,
                               k=k, save_dir=save_dir_combined_neighbor, file_name='stabilities_combined')

    if method == "neighbor":
        get_stability_neighbors(models=models, words_path=keywords_path,
                                k=k, save_dir=save_dir_combined_neighbor, file_name='stabilities_neighbor')

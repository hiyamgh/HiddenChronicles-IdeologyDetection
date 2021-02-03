import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
import random
import math
import logging
from bias.utilities import *


def _random_permutation(iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def word_association_with_attribute_precomputed_sims(w, A, B):
    return np.mean([cossim(w, a) for a in A]) - np.mean([cossim(w, b) for b in B])


def differential_association_precomputed_sims(T1, T2, A1, A2):
    return np.sum([word_association_with_attribute_precomputed_sims(t1, A1, A2) for t1 in T1]) - np.sum([word_association_with_attribute_precomputed_sims(t2, A1, A2) for t2 in T2])


def weat_effect_size_precomputed_sims(T1, T2, A1, A2):
    return (
             np.mean([word_association_with_attribute_precomputed_sims(t1, A1, A2) for t1 in T1]) -
             np.mean([word_association_with_attribute_precomputed_sims(t2, A1, A2) for t2 in T2])
           ) / np.std([word_association_with_attribute_precomputed_sims(w, A1, A2) for w in T1 + T2])


def weat_p_value_precomputed_sims(T1, T2, A1, A2, model, sample):
    logging.info("Calculating p value ... ")
    print('caclulating p value ...')
    size_of_permutation = min(len(T1), len(T2))
    T1_T2 = T1 + T2
    observed_test_stats_over_permutations = []
    total_possible_permutations = math.factorial(len(T1_T2)) / math.factorial(size_of_permutation) / math.factorial((len(T1_T2)-size_of_permutation))
    logging.info("Number of possible permutations: %d", total_possible_permutations)
    if not sample or sample >= total_possible_permutations:
      permutations = combinations(T1_T2, size_of_permutation)
    else:
      logging.info("Computing randomly first %d permutations", sample)
      print("Computing randomly first %d permutations", sample)
      permutations = set()
      while len(permutations) < sample:
        permutations.add(tuple(sorted(_random_permutation(T1_T2, size_of_permutation))))

    A1_vecs = model.wv[A1]
    A2_vecs = model.wv[A2]

    for Xi in permutations:
      Yi = filterfalse(lambda w: w in Xi, T1_T2)

      Xi_vecs = model.wv[Xi]
      Yi_vecs = [model.wv[y] for y in Yi]

      observed_test_stats_over_permutations.append(differential_association_precomputed_sims(Xi_vecs, Yi_vecs,
                                                                                             A1_vecs, A2_vecs))
      if len(observed_test_stats_over_permutations) % 100000 == 0:
        logging.info("Iteration %s finished", str(len(observed_test_stats_over_permutations)))

    T1_vecs = model.wv[T1]
    T2_vecs = model.wv[T2]
    unperturbed = differential_association_precomputed_sims(T1_vecs, T2_vecs, A1_vecs, A2_vecs)

    is_over = np.array([o > unperturbed for o in observed_test_stats_over_permutations])
    return is_over.sum() / is_over.size


def embedding_coherence_test(word2vec_currmodel, target_1, target_2, attributes):
    repres1 = get_mean_vector(word2vec_currmodel, target_1)
    repres2 = get_mean_vector(word2vec_currmodel, target_2)

    sims_first = []
    sims_second = []
    for a in attributes:
        # sims_first.append(np.dot(avg_first, vec_a) / (np.linalg.norm(avg_first) * np.linalg.norm(vec_a)))
        # sims_second.append(np.dot(avg_second, vec_a) / (np.linalg.norm(avg_second) * np.linalg.norm(vec_a)))
        vec_a = word2vec_currmodel.wv[a]
        sims_first.append(cossim(repres1, vec_a))
        sims_second.append(cossim(repres2, vec_a))

    return stats.spearmanr(sims_first, sims_second)

def calculate_weat_bias(target_list1, target_list2, attr_list1, attr_list2, s_year, e_year,
                        word2vec_models_path, distype='norm', topKneighbs=3):
    '''
    Run the WEAT test for differential association between two
    sets of target words and two sets of attributes.
    RETURNS:
        (d, e, p). A tuple of floats, where d is the WEAT Test statistic,
        e is the effect size, and p is the one-sided p-value measuring the
        (un)likeliness of the null hypothesis (which is that there is no
        difference in association between the two target word sets and
        the attributes).
        If e is large and p small, then differences in the model between
        the attribute word sets match differences between the targets.
    '''

    word2vec_models_dict = load_models(start_year=s_year, end_year=e_year, archive_path=word2vec_models_path)
    for year in range(s_year, e_year + 1):
        word2vec_currmodel = word2vec_models_dict[year]

        print('------------------------------------ YEAR: {} ------------------------------------'.format(year))

        target_list1 = check_terms(target_list1, word2vec_currmodel)
        target_list2 = check_terms(target_list2, word2vec_currmodel)

        attr_list1 = check_terms(attr_list1, word2vec_currmodel)
        attr_list2 = check_terms(attr_list2, word2vec_currmodel)

        if target_list1 == [] or target_list2 == [] or attr_list1 == [] or attr_list2 == []:
            raise ValueError('one of the lists is empty')

        print('checked terms ...')

        if target_list1 and target_list2:
            if len(target_list1) == len(target_list2):
                pass
            elif len(target_list1) < len(target_list2):
                target_list1 = apply_augmentation(larger_list=target_list2, smaller_list=target_list1,
                                                word2vec_currmodel=word2vec_currmodel,
                                                topK=topKneighbs)
            else:
                target_list2 = apply_augmentation(larger_list=target_list1, smaller_list=target_list2,
                                                word2vec_currmodel=word2vec_currmodel,
                                                topK=topKneighbs)
        if attr_list1 and attr_list2:
            if len(attr_list1) == len(attr_list2):
                pass
            elif len(attr_list1) < len(attr_list2):
                attr_list1 = apply_augmentation(larger_list=attr_list2, smaller_list=attr_list1,
                                                  word2vec_currmodel=word2vec_currmodel,
                                                  topK=topKneighbs)
            else:
                attr_list2 = apply_augmentation(larger_list=attr_list1, smaller_list=attr_list2,
                                                word2vec_currmodel=word2vec_currmodel,
                                                topK=topKneighbs)

        print('target_list1 populated: {}'.format(target_list1))
        print('target_list2 populated: {}'.format(target_list2))
        print('attr_list1 populated: {}'.format(attr_list1))
        print('attr_list2 populated: {}'.format(attr_list2))

        t1_vecs = [word2vec_currmodel.wv[t1] for t1 in target_list1]
        t2_vecs = [word2vec_currmodel.wv[t2] for t2 in target_list2]
        a1_vecs = [word2vec_currmodel.wv[a1] for a1 in attr_list1]
        a2_vecs = [word2vec_currmodel.wv[a2] for a2 in attr_list2]

        test_statistic = differential_association_precomputed_sims(t1_vecs, t2_vecs, a1_vecs, a2_vecs)
        effect_size = weat_effect_size_precomputed_sims(t1_vecs, t2_vecs, a1_vecs, a2_vecs)
        p = weat_p_value_precomputed_sims(target_list1, target_list2, attr_list1,
                                          attr_list2, model=word2vec_currmodel,
                                          sample=1000)

        print('test statistic: {}'.format(test_statistic))
        print('effect size: {}'.format(effect_size))
        print('p-value: {}'.format(p))


def calculate_weat_bias_decade_level(target_list1, target_list2, attr_list1, attr_list2,
                        decades_path, distype='norm', topKneighbs=3):

    weat_decades = pd.DataFrame()
    decades = get_decades(decades_path=decades_path)
    decades = sorted(decades, key=lambda x: x[0])
    for decade in decades:
        s_year = decade[0]
        e_year = decade[1]

        word2vec_currmodel = load_model_decade_level(decades_path, s_year, e_year)

        print('------------------------------------ DECADE: {}-{} ------------------------------------'.format(s_year, e_year))

        target_list1 = check_terms(target_list1, word2vec_currmodel)
        target_list2 = check_terms(target_list2, word2vec_currmodel)

        attr_list1 = check_terms(attr_list1, word2vec_currmodel)
        attr_list2 = check_terms(attr_list2, word2vec_currmodel)

        if target_list1 == [] or target_list2 == [] or attr_list1 == [] or attr_list2 == []:
            raise ValueError('one of the lists is empty')

        print('checked terms ...')

        if target_list1 and target_list2:
            if len(target_list1) == len(target_list2):
                pass
            elif len(target_list1) < len(target_list2):
                target_list1 = apply_augmentation(larger_list=target_list2, smaller_list=target_list1,
                                                  word2vec_currmodel=word2vec_currmodel,
                                                  topK=topKneighbs)
            else:
                target_list2 = apply_augmentation(larger_list=target_list1, smaller_list=target_list2,
                                                  word2vec_currmodel=word2vec_currmodel,
                                                  topK=topKneighbs)
        if attr_list1 and attr_list2:
            if len(attr_list1) == len(attr_list2):
                pass
            elif len(attr_list1) < len(attr_list2):
                attr_list1 = apply_augmentation(larger_list=attr_list2, smaller_list=attr_list1,
                                                word2vec_currmodel=word2vec_currmodel,
                                                topK=topKneighbs)
            else:
                attr_list2 = apply_augmentation(larger_list=attr_list1, smaller_list=attr_list2,
                                                word2vec_currmodel=word2vec_currmodel,
                                                topK=topKneighbs)

        print('target_list1 populated: {}'.format(target_list1))
        print('target_list2 populated: {}'.format(target_list2))
        print('attr_list1 populated: {}'.format(attr_list1))
        print('attr_list2 populated: {}'.format(attr_list2))

        t1_vecs = [word2vec_currmodel.wv[t1] for t1 in target_list1]
        t2_vecs = [word2vec_currmodel.wv[t2] for t2 in target_list2]
        a1_vecs = [word2vec_currmodel.wv[a1] for a1 in attr_list1]
        a2_vecs = [word2vec_currmodel.wv[a2] for a2 in attr_list2]

        test_statistic = differential_association_precomputed_sims(t1_vecs, t2_vecs, a1_vecs, a2_vecs)
        effect_size = weat_effect_size_precomputed_sims(t1_vecs, t2_vecs, a1_vecs, a2_vecs)
        p = weat_p_value_precomputed_sims(target_list1, target_list2, attr_list1,
                                          attr_list2, model=word2vec_currmodel,
                                          sample=1000)

        print('test statistic: {}'.format(test_statistic))
        print('effect size: {}'.format(effect_size))
        print('p-value: {}'.format(p))

        weat_decades['{}-{}'.format(s_year, e_year)] = [effect_size]

    weat_decades.to_csv('weat_decades.csv', index=False)

def bias_analogy_test(word2vec_currmodel, target_1, target_2, attributes_1, attributes_2):

    to_rmv = [x for x in attributes_1 if x in attributes_2]
    for x in to_rmv:
        attributes_1.remove(x)
        attributes_2.remove(x)

    if len(attributes_1) != len(attributes_2):
        min_len = min(len(attributes_1), len(attributes_2))
        attributes_1 = attributes_1[:min_len]
        attributes_2 = attributes_2[:min_len]
    print(attributes_1)
    print(attributes_2)

    atts_paired = []
    for a1 in attributes_1:
        for a2 in attributes_2:
            atts_paired.append((a1, a2))

    tmp_vocab = list(set(target_1 + target_2 + attributes_1 + attributes_2))
    dicto = []
    matrix = []
    for w in tmp_vocab:
        if w in word2vec_currmodel.wv:
            matrix.append(word2vec_currmodel.wv[w])
            dicto.append(w)

    vecs = np.array(matrix)
    vocab = {dicto[i]: i for i in range(len(dicto))}

    eq_pairs = []
    for t1 in target_1:
        for t2 in target_2:
            eq_pairs.append((t1, t2))

    for pair in eq_pairs:
        t1 = pair[0]
        t2 = pair[1]
        vec_t1 = vecs[vocab[t1]]
        vec_t2 = vecs[vocab[t2]]

        biased = []
        totals = []
        for a1, a2 in atts_paired:
            vec_a1 = vecs[vocab[a1]]
            vec_a2 = vecs[vocab[a2]]

            diff_vec = vec_t1 - vec_t2

            query_1 = diff_vec + vec_a2
            query_2 = vec_a1 - diff_vec

            sims_q1 = np.sum(np.square(vecs - query_1), axis=1)
            sorted_q1 = np.argsort(sims_q1)
            ind = np.where(sorted_q1 == vocab[a1])[0][0]
            other_att_2 = [x for x in attributes_2 if x != a2]
            indices_other = [np.where(sorted_q1 == vocab[x])[0][0] for x in other_att_2]
            num_bias = [x for x in indices_other if ind < x]
            biased.append(len(num_bias))
            totals.append(len(indices_other))

            sims_q2 = np.sum(np.square(vecs - query_2), axis=1)
            sorted_q2 = np.argsort(sims_q2)
            ind = np.where(sorted_q2 == vocab[a2])[0][0]
            other_att_1 = [x for x in attributes_1 if x != a1]
            indices_other = [np.where(sorted_q2 == vocab[x])[0][0] for x in other_att_1]
            num_bias = [x for x in indices_other if ind < x]
            biased.append(len(num_bias))
            totals.append(len(indices_other))

    return sum(biased) / sum(totals)


def eval_k_means(word2vec_currmodel, t1_list, t2_list):
    '''
      Implicit bias evaluation
      :param t1_list: target terms of T1 (list)
      :param t2_list: target terms of T1 (list)
      :param vocab: word2index dict
      :param vecs: index2vector matrix
      :return: avg score over 50 runs
    '''

    # lista = t1_list + t2_list
    # word_vecs = []
    # for l in lista:
    #     if l in vocab:
    #         word_vecs.append(vecs[vocab[l]])
    #     else:
    #         print(l + " not in vocab!")

    lista = t1_list + t2_list
    word_vecs = [v for v in word2vec_currmodel.wv[lista]]
    vecs_to_cluster = word_vecs
    golds1 = [1]*len(t1_list) + [0] * len(t2_list)
    golds2 = [0]*len(t1_list) + [1] * len(t2_list)
    items = list(zip(vecs_to_cluster, golds1, golds2))

    scores = []
    for i in range(50):
        random.shuffle(items)
        kmeans = KMeans(n_clusters=2, random_state=0, init= 'k-means++').fit(np.array([x[0] for x in items]))
        preds = kmeans.labels_

        acc1 = len([i for i in range(len(preds)) if preds[i] == items[i][1]]) / len(preds)
        acc2 = len([i for i in range(len(preds)) if preds[i] == items[i][2]]) / len(preds)
        scores.append(max(acc1, acc2))

    return sum(scores) / len(scores)


# def embedding_coherence_test(word2vec_currmodel, target_1, target_2, attributes):
#   """
#   target_1 and target_2 here could be words (will convert them to vectors inside get_mean_vector method used below)
#   Explicit bias evaluation
#   :param vecs: index2vec vector matrix
#   :param vocab: term2index dict
#   :param target_1: list of t1 terms
#   :param target_2: list of t2 terms
#   :param attributes: list of attributes
#   :return: spearman correlation
#   """
#
#   # get the representative vector from each group, which is the mean of the vectors of each word in the wordlist
#   repres1 = get_mean_vector(word2vec_currmodel, target_1)
#   repres2 = get_mean_vector(word2vec_currmodel, target_2)
#
#   sims_first = []
#   sims_second = []
#   for a in attributes:
#     if a in word2vec_currmodel.vw:
#         # sims_first.append(np.dot(avg_first, vec_a) / (np.linalg.norm(avg_first) * np.linalg.norm(vec_a)))
#         # sims_second.append(np.dot(avg_second, vec_a) / (np.linalg.norm(avg_second) * np.linalg.norm(vec_a)))
#       vec_a = word2vec_currmodel.wv[a]
#       sims_first.append(cossim(repres1, vec_a))
#       sims_second.append(cossim(repres2, vec_a))
#   return stats.spearmanr(sims_first, sims_second)

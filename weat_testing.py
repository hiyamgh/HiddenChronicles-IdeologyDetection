import numpy as np
import random
from itertools import *
from gensim.models import Word2Vec
alphabet = list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B))
diactitics = list(range(0x064B, 0x0653))

alphabet = [chr(x) for x in alphabet]
diactitics = [chr(x) for x in diactitics]


war_key_terms = [
'إداري',
'احتجاز',
'الحدود',
'تجاوز',
'طريق',
'إغلاق',
'جماعي',
'عقاب',
'عرقي',
'تطهير',
'أخضر',
'خط',
'الانتفاضة',
'مسدود',
'فلسطيني',
'إقليم',
'السلطة',
'لاجئ',
'مستوطنة',
'انفصال',
'حائط',
'صهيونية',
]

location_palestine = [
# 'الضفة الغربية',
# 'الخليل',
'غزه',
# 'رام الله',
# 'بيت',
'الاقصي',
]


# location_israel = ['Israel', 'Jerusalem', 'Beit', 'Tel Aviv']
location_israel = [
'إسرائيل',
'بيت المقدس',
'بيت',
'تل أبيب',
]


violence = [
'قتل',
'الهجمات',
'عنف',
'طعن',
# 'اطلاق النار',
# 'جرحى',
'اشتباكات',
'الاحتلال',
]

amity = [
'سلام',
'هدنة',
'وحدة',
'الصداقة',
]

class WEATTest(object):
    """
    Perform WEAT (Word Embedding Association Test) bias tests on a language model.
    Follows from Caliskan et al 2017 (10.1126/science.aal4230).
    """

    def __init__(self, model):
        # """Setup a Word Embedding Association Test for a given spaCy language model.
        #
        # EXAMPLE:
        #     >>> nlp = spacy.load('en_core_web_md')
        #     >>> test = WEATTest(nlp)
        #     >>> test.run_test(WEATTest.instruments, WEATTest.weapon, WEATTest.pleasant, WEATTest.unpleasant)
        # """
        self.model = model

    def word_association_with_attribute(self, w, A, B):
        # return np.mean([w.similarity(a) for a in A]) - np.mean([w.similarity(b) for b in B])
        return np.mean([self.model.similarity(w, a) for a in A]) - np.mean([self.model.similarity(w, b) for b in B])

    def differential_assoication(self, X, Y, A, B):
        return np.sum([self.word_association_with_attribute(x, A, B) for x in X]) - np.sum(
            [self.word_association_with_attribute(y, A, B) for y in Y])

    def weat_effect_size(self, X, Y, A, B):
        return (
                       np.mean([self.word_association_with_attribute(x, A, B) for x in X]) -
                       np.mean([self.word_association_with_attribute(y, A, B) for y in Y])
               ) / np.std([self.word_association_with_attribute(w, A, B) for w in X + Y])

    def _random_permutation(self, iterable, r=None):
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        return tuple(random.sample(pool, r))

    def weat_p_value(self, X, Y, A, B, sample):
        size_of_permutation = min(len(X), len(Y))
        X_Y = X + Y
        observed_test_stats_over_permutations = []

        if not sample:
            permutations = combinations(X_Y, size_of_permutation)
        else:
            permutations = [self._random_permutation(X_Y, size_of_permutation) for s in range(sample)]

        for Xi in permutations:
            Yi = filterfalse(lambda w: w in Xi, X_Y)
            observed_test_stats_over_permutations.append(self.differential_assoication(Xi, Yi, A, B))

        unperturbed = self.differential_assoication(X, Y, A, B)
        is_over = np.array([o > unperturbed for o in observed_test_stats_over_permutations])
        return is_over.sum() / is_over.size

    def weat_stats(self, X, Y, A, B, sample_p=None):
        test_statistic = self.differential_assoication(X, Y, A, B)
        effect_size = self.weat_effect_size(X, Y, A, B)
        p = self.weat_p_value(X, Y, A, B, sample=sample_p)
        return test_statistic, effect_size, p

    def run_test(self, target_1, target_2, attributes_1, attributes_2, sample_p=None):
        """Run the WEAT test for differential association between two
        sets of target words and two seats of attributes.

        # EXAMPLE:
        #     >>> test.run_test(WEATTest.instruments, WEATTest.weapon, WEATTest.pleasant, WEATTest.unpleasant)
        #     >>> test.run_test(a, b, c, d, sample_p=1000) # use 1000 permutations for p-value calculation
        #     >>> test.run_test(a, b, c, d, sample_p=None) # use all possible permutations for p-value calculation

        RETURNS:
            (d, e, p). A tuple of floats, where d is the WEAT Test statistic,
            e is the effect size, and p is the one-sided p-value measuring the
            (un)likeliness of the null hypothesis (which is that there is no
            difference in association between the two target word sets and
            the attributes).

            If e is large and p small, then differences in the model between
            the attribute word sets match differences between the targets.
        """
        # X = [self.model(w) for w in target_1]
        # Y = [self.model(w) for w in target_2]
        # A = [self.model(w) for w in attributes_1]
        # B = [self.model(w) for w in attributes_2]
        return self.weat_stats(target_1, target_2, attributes_1, attributes_2, sample_p)


def edits1(word):
    "All edits that are one edit away from `word`."
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in alphabet]
    inserts    = [L + c + R               for L, R in splits for c in alphabet]
    return list(set(deletes + transposes + replaces + inserts))


def edits2(word):
    ''' All edits that are two edit distances away from `word` '''
    return [e2 for e1 in edits1(word) for e2 in edits1(e1)]

# print(edits1('hiyam'))


# def populate_list(target_terms, topK):
#     for t in target_terms:
#         neighbors = word_vectors.most_similar(positive=[t], topn=topK)
#         target_terms = target_terms + [n[0] for n in neighbors]
#     return target_terms


# if t not in vocab:
#     print('Term {} not in vocab, will try to search for its edits'.format(t))
#     possibilities = edits1(t, arabic_alphabet=alphabet)
#     if any(possibilities) in vocab:
#         possibles = [p for p in possibilities if p in vocab]
#         # return possibles
#         population_missing = population_missing + possibles
#     else:
#         raise ValueError('Neither {} nor its edits are in the vocabulary'.format(t))


def get_edits_missing(t, vocab):
    possibilities = edits1(t)
    if any(possibilities) in vocab:
        print('got the words that are 1 edit distance away from {}'.format(t))
        possibilities = [p for p in possibilities if p in vocab]
        return possibilities
    else:
        possibilities = edits2(t)
        if any(possibilities) in vocab:
            print('got the words that are 2 edit distances away from {}'.format(t))
            return possibilities
        else:
            return -1


def populate_list(target_terms, vocab, topK):
    populated = []
    for t in target_terms:
        if t in vocab:
            tedit1 = edits1(t)
            tedit1 = [te for te in tedit1 if te in vocab]

            tedit1sorted = {}
            for te in tedit1:
                tedit1sorted[te] = model.similarity(t, te)
            tedit1sorted = {k: v for k, v in sorted(tedit1sorted.items(), key=lambda item: item[1], reverse=True)}
            top_edits = dict(islice(tedit1sorted.items(), topK))
            print('\n top neighbors for {}:'.format(t))
            for k, v in top_edits.items():
                print('{}: {}'.format(k, v))
            populated = populated + list(top_edits.keys())
        else:
            if t == 'عنف':
                print()
            possible_from_missing = get_edits_missing(t, vocab)
            if possible_from_missing != -1:
                print('\n top neighbors for {} (did not find t, getting its edits that exist in vocab):'.format(t))
                print('edits that exist: {}'.format(possible_from_missing))
                populated = populated + possible_from_missing
            else:
                print('skipping {} because neither it nor its edits are in the vocab'.format(t))

    return populated


if __name__ == '__main__':
    # model = Word2Vec.load('trained_models2/word2vec')
    model = Word2Vec.load('D:/word2vec/hayat/embeddings/word2vec_1950')
    word_vectors = model.wv

    # populate_list(location_palestine, word_vectors, topK=5)
    print(violence)
    violence_populated = list(set(populate_list(violence, word_vectors, topK=3)))
    print('violence populated: {}'.format(violence_populated))
    # print(amity)
    amity_populated = list(set(populate_list(amity, word_vectors, topK=3)))
    print('amity populated: {}'.format(amity_populated))
    # location_palestine = [t for t in location_palestine if t in word_vectors.vocab]
    # location_palestine = populate_list(location_palestine, topK=3)
    # print(location_palestine)
    #
    # location_israel = [t for t in location_israel if t in word_vectors.vocab]
    # location_israel = populate_list(location_israel, topK=3)
    # print(location_israel)
    #
    # violence = [t for t in violence if t in word_vectors.vocab]
    # # violence = populate_list(violence, topK=3)
    #
    # amity = [t for t in amity if t in word_vectors.vocab]
    # # amity = populate_list(amity, topK=3)
    #
    # weatTest = WEATTest(model)
    # test_statistic, effect_size, p = weatTest.run_test(location_palestine, location_israel, violence, amity)
    # print('test statistic: {}'.format(test_statistic))
    # print('effect size: {}'.format(effect_size))
    # print('p-value: {}'.format(p))


# C:\Users\96171\AppData\Local\Programs\Python\Python36\python.exe C:/Users/96171/Desktop/newspapers_mining/terms_targets_attributes.py
# [
# الاقصي,  غزه, الضفه,
# اسراءيل تلابيب
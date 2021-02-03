import os, pickle
import numpy as np
from gensim.models import Word2Vec
from bias.utilities import get_edits_missing
from smart_open import open
from gensim.models import translation_matrix
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import getpass
import time
import itertools
import multiprocessing


# nltk for getting stopwords
from nltk.corpus import stopwords
stopwords_list = stopwords.words('arabic')
from bias.utilities import cossim, file_to_list, check_terms


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_children(self, obj):
        self.children = []
        for o in obj:
            self.children.append(Node(o))


class GraphNeighbourBased:

    def __init__(self, model1, model2, t, topK=100):
        v1 = list(model1.wv.vocab.keys())
        v2 = list(model2.wv.vocab.keys())
        # the intersection of the vocabs of two embedding spaces
        self.v = list(set(v1) & set(v2))
        # number of iterations - Algorithm 1 in the paper
        self.t = t
        # two embedding spaces
        self.model1, self.model2 = model1, model2
        # top N neighbours to include in Algorithm 1
        self.topK = topK
        # vector containing stability of each word in vocab
        svals = [1] * len(self.v)
        # dictionary containing stability values of each word in v
        self.stabilities = dict(zip(self.v, svals))
        # after initialization build the graph
        self._build_graph()

    def _build_graph(self):
        t1 = time.time()
        self.root_nodes = [Node(w) for w in self.v]
        self.graph1 = [Node(w) for w in self.v]
        self.graph2 = [Node(w) for w in self.v]
        # keep adding neighbours until depth=t
        for iter in range(self.t):
            print('building graph: iteration {}'.format(iter))
            if iter == 0:
                p1 = multiprocessing.Process(target=self._add_first_children2leaves_graph1)
                p2 = multiprocessing.Process(target=self._add_first_children2leaves_graph2)

                p1.start()
                p2.start()

                # wait until process 1 is finished
                p1.join()
                # wait until process 2 is finished
                p2.join()

                # for i, w in enumerate(self.graph1):
                #     nei1 = self.model1.wv.most_similar(positive=[w.data], topn=self.topK)
                #     self.graph1[i].add_children([n for n, _ in nei1])
                #
                # for i, w in enumerate(self.graph2):
                #     nei2 = self.model2.wv.most_similar(positive=[w.data], topn=self.topK)
                #     self.graph2[i].add_children([n for n, _ in nei2])
            else:
                p1 = multiprocessing.Process(target=self._add_children2leaves_graph1)
                p2 = multiprocessing.Process(target=self.add_children2leaves_graph2)

                p1.start()
                p2.start()

                # wait until process 1 is finished
                p1.join()
                # wait until process 2 is finished
                p2.join()
                # for node in self.graph1:
                #     leaves = [leaf for leaf in self.get_leaves(node)]
                #     for l in leaves:
                #         neighs = self.model1.wv.most_similar(positive=[l.data], topn=self.topK)
                #         l.add_children([n for n, _ in neighs])
                # for node in self.graph2:
                #     leaves = [leaf for leaf in self.get_leaves(node)]
                #     for l in leaves:
                #         neighs = self.model2.wv.most_similar(positive=[l.data], topn=self.topK)
                #         l.add_children([n for n, _ in neighs])
        print('done building graph ...')
        t2 = time.time()
        print('building graph took: {}'.format((t2-t1)/60))
        print('saving graphs ...')
        with open("graph1.dat", "wb") as f:
            pickle.dump(self.graph1, f)
        with open("graph2.dat", "wb") as f:
            pickle.dump(self.graph2, f)
        print('done saving graphs')

    def _add_first_children2leaves_graph1(self):
        for i, w in enumerate(self.graph1):
            nei1 = self.model1.wv.most_similar(positive=[w.data], topn=self.topK)
            self.graph1[i].add_children([n for n, _ in nei1])

    def _add_first_children2leaves_graph2(self):
        for i, w in enumerate(self.graph2):
            nei2 = self.model2.wv.most_similar(positive=[w.data], topn=self.topK)
            self.graph2[i].add_children([n for n, _ in nei2])

    def _add_children2leaves_graph1(self):
        for node in self.graph1:
            leaves = [leaf for leaf in self.get_leaves(node)]
            for l in leaves:
                neighs = self.model1.wv.most_similar(positive=[l.data], topn=self.topK)
                l.add_children([n for n, _ in neighs])

    def add_children2leaves_graph2(self):
        for node in self.graph2:
            leaves = [leaf for leaf in self.get_leaves(node)]
            for l in leaves:
                neighs = self.model2.wv.most_similar(positive=[l.data], topn=self.topK)
                l.add_children([n for n, _ in neighs])

    def get_leaves(self, w):
        if not w.children:
            print('leaf: {}'.format(w.data))
            yield w.data

        else:
            for i in range(len(w.children)):
                print('processing child: {}'.format(w.children[i].data))
                yield from self.get_leaves(w.children[i])
        # result = []
        # if not w.children:
        #     result = w
        # else:
        #     result.extend([self.get_leaves(w.children[i]) for i in range(len(w.children))])
        # return result

    def get_neighbours(self, w, iter):
        ''' at first iteration, only direct neighbours contribute to the stability of word
            w. At iteration t=k, the indirect neighbours accessible by k edges in the graph contribute
            to the stability of the word w.
            iter: depth at which neighbours must be
         '''
        if iter == 0:
            for child in w.children:
                yield child
        else:
            for i in range(len(w.children)):
                yield from self.get_neighbours(w.children[i], iter-1)

    def build_stability(self):
        # either graph 1 or graph 2, they both have same roots (the difference is only in the children)
        for t in range(self.t):
            stabilities = np.array(self.stabilities)
            for i in range(len(self.v)):
                w1 = self.graph1[i]
                w2 = self.graph2[i]
                # get all neighbours of w in second embedding space
                neighbours1 = [nei for nei in self.get_neighbours(w1, t)]
                neighbours2 = [nei for nei in self.get_neighbours(w2, t)]
                # calculate cosine similarity between w in first embedding space
                # and neighbour in second embedding space
                summation01, summation10 = 0, 0
                for n2 in neighbours2:
                    v1 = self.model1.wv[w1]
                    v2 = self.model1.wv[n2.data]
                    summation01 += cossim(v1, v2) * self.stabilities[n2.data]

                sim01 = summation01/len(neighbours2)

                for n1 in neighbours1:
                    v1 = self.model2.wv[w2]
                    v2 = self.model2.wv[n1.data]
                    summation10 += cossim(v1, v2) * self.stabilities[n1.data]

                sim10 = summation10/len(neighbours1)
                snei = (sim01 + sim10) / 2
                stabilities[i] = snei

            stabilities = [(float(s) - min(stabilities)) / (max(stabilities) - min(stabilities)) for s in stabilities]


# def get_leaves(w):
#     if not w.children:
#         print('leaf: {}'.format(w.data))
#         yield w.data
#
#     else:
#         for i in range(len(w.children)):
#             print('processing child: {}'.format(w.children[i].data))
#             yield from get_leaves(w.children[i])


def my_generator0(n):
    for i in range(n):
        yield i
        # if i >= 5:
        #     return


if __name__ == '__main__':
    # print([leaf for leaf in my_generator0(10)])
    # n1 = Node(10)
    # n1.add_children([11,12,13])
    # n1.children[0].add_children([5,6,7])
    # n1.children[0].children[2].add_children([20,30,40])
    # # leaves = get_leaves(n1)
    # print('final result is: {}'.format([leaf for leaf in get_leaves(n1)]))

    if getpass.getuser() == '96171':
        model1 = Word2Vec.load('E:/newspapers/word2vec/nahar/embeddings/word2vec_1933')
        model2 = Word2Vec.load('E:/newspapers/word2vec/nahar/embeddings/word2vec_1976')
    else:
        model1 = Word2Vec.load('D:/word2vec/nahar/embeddings/word2vec_1933')
        model2 = Word2Vec.load('D:/word2vec/nahar/embeddings/word2vec_1976')
    graph_nei = GraphNeighbourBased(model1=model1, model2=model2, t=5, topK=10)

# class NeighboursGraph:
#
#     def __init__(self, model1, model2, T, V):
#         """
#         :param model1: first embedding space
#         :param model2: second embedding space
#         :param T: the number of iterations
#         :param V: the intersection of vocabularies
#         """
#         self.model1 = model1
#         self.model2 = model2
#         self.T = T
#         self.V = V
#
#     def build_graph(self):
#         # at first iteration only direct neighbours contribute to the stability of the
#         # word
#
#         # consider each word in an embedding space as a node and its neighbours are its closest
#         # nodes to it, measured by cosine similarity
#
#         # search for direct neighbours
#         self.N1 = np.zeros((len(self.V), self.T + 1))
#         self.N2 = np.zeros((len(self.V), self.T + 1))
#
#         self.N1[:, 0] = self.V
#         self.N2[:, 0] = self.V
#
#         for iter in range(1, self.T + 1):
#             self.N1[:, iter] = [self.model1.wv.most_similar(positive=[w], topn=iter)[-1] for w in self.V]
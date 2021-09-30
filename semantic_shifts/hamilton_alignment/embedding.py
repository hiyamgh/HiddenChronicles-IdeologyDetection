from gensim.models import Word2Vec
import numpy as np
import heapq
from sklearn.preprocessing import normalize

class Embedding:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """

    def __init__(self, vecs, normalize=True, **kwargs):
        self.wv = vecs
        self.m = vecs.vectors # matrix of vectors
        self.dim = self.m.shape[1] # diemnsion of vectors
        self.iw = vecs.index2entity # mapping of index to word
        self.wi = {w: i for i, w in enumerate(self.iw)} # mapping of word to index
        if normalize:
            self.normalize()

    def __getitem__(self, key):
        if self.oov(key):
            raise KeyError
        else:
            return self.represent(key)

    def __iter__(self):
        return self.iw.__iter__()

    def __contains__(self, key):
        return not self.oov(key)

    @classmethod
    def load(cls, path, normalize=True, add_context=False, **kwargs):
        ''' Changes to original .load() method implemented by Hamilton -- to support gensim '''
        # mat = np.load(path + "-w.npy", mmap_mode="c")
        # if add_context:
        #     mat += np.load(path + "-c.npy", mmap_mode="c")
        # iw = load_pickle(path + "-vocab.pkl")
        # return cls(mat, iw, normalize)
        model = Word2Vec.load(path)
        return cls(model.wv, normalize)

    def get_subembed(self, word_list, **kwargs):
        ''' Changes to original .get_sumembed() method implemented by Hamilton -- to support gensim '''
        word_list = [word for word in word_list if not self.oov(word)]
        # keep_indices = [self.wi[word] for word in word_list]
        # return Embedding(self.m[keep_indices, :], word_list, normalize=False)
        # return Embedding(self.m[keep_indices, :], normalize=False)
        return Embedding(self.restrict_w2v(self.wv, word_list))

    # def reindex(self, word_list, **kwargs):
    #     new_mat = np.empty((len(word_list), self.m.shape[1]))
    #     valid_words = set(self.iw)
    #     for i, word in enumerate(word_list):
    #         if word in valid_words:
    #             new_mat[i, :] = self.represent(word)
    #         else:
    #             new_mat[i, :] = 0
    #     return Embedding(new_mat, word_list, normalize=False)

    def get_neighbourhood_embed(self, w, n=1000):
        # neighbours = self.closest(w, n=n)
        neighbours_sims = self.wv.most_similar(w, n=n)
        neighbours = [t[0] for t in neighbours_sims]
        return Embedding(self.restrict_w2v(self.wv, neighbours))
        # keep_indices = [self.wi[neighbour] for _, neighbour in neighbours]
        # new_mat = self.m[keep_indices, :]
        # return Embedding(new_mat)

    def restrict_w2v(self, w2v, restricted_word_set):
        new_vectors = []
        new_vocab = {}
        new_index2entity = []
        new_vectors_norm = []

        for i in range(len(w2v.vocab)):
            word = w2v.index2entity[i]
            vec = w2v.vectors[i]
            vocab = w2v.vocab[word]
            vec_norm = w2v.vectors_norm[i]
            if word in restricted_word_set:
                vocab.index = len(new_index2entity)
                new_index2entity.append(word)
                new_vocab[word] = vocab
                new_vectors.append(vec)
                new_vectors_norm.append(vec_norm)

        w2v.vocab = new_vocab
        w2v.vectors = np.array(new_vectors)
        w2v.index2entity = np.array(new_index2entity)
        w2v.index2word = np.array(new_index2entity)
        w2v.vectors_norm = np.array(new_vectors_norm)

        return w2v

    def normalize(self):
       normalize(self.m, copy=False)

    def oov(self, w):
        # return not (w in self.wi)
        return not (w in self.iw)

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            print("OOV: ", w)
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        sim = self.represent(w1).dot(self.represent(w2))
        return sim

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, zip(scores, self.iw))


if __name__ == '__main__':
    model = Word2Vec.load('F:/newspapers/word2vec/nahar/embeddings/word2vec_2005')
    e = Embedding(model.wv)
    print()

    # get word from index: model.wv.index2entity[i]
    # matrix of embeddings: model.wv.vectors
    # propoerties about word: model.wv.vocab:
                            # * count
                            # index
                            # sample_int
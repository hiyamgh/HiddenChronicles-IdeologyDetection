import numpy as np
from embedding import Embedding
from gensim.models.word2vec import Vocab
""" Some methods for aligning embeddings spaces """

def explicit_intersection_align(embed1, embed2, restrict_context=True):
    common_vocab = filter(set(embed1.ic).__contains__, embed2.ic) 
    return embed1.get_subembed(common_vocab, restrict_context=restrict_context), embed2.get_subembed(common_vocab, restrict_context=restrict_context)
    
def intersection_align(embed1, embed2, post_normalize=True):
    """ 
        Get the intersection of two embeddings.
        Returns embeddings with common vocabulary and indices.
    """
    common_vocab = filter(set(embed1.iw).__contains__, embed2.iw) 
    newvecs1 = np.empty((len(common_vocab), embed1.m.shape[1]))
    newvecs2 = np.empty((len(common_vocab), embed2.m.shape[1]))
    # for i in xrange(len(common_vocab)):
    for i in range(len(common_vocab)):
        newvecs1[i] = embed1.m[embed1.wi[common_vocab[i]]]
        newvecs2[i] = embed2.m[embed2.wi[common_vocab[i]]]
    return Embedding(newvecs1, common_vocab, normalize=post_normalize), Embedding(newvecs2, common_vocab, normalize=post_normalize)

def get_procrustes_mat(base_embed, other_embed):
    in_base_embed, in_other_embed = intersection_align(base_embed, other_embed, post_normalize=False)
    base_vecs = in_base_embed.m
    other_vecs = in_other_embed.m
    m = other_vecs.T.dot(base_vecs)
    u, _, v = np.linalg.svd(m) 
    ortho = u.dot(v)
    return ortho

def smart_procrustes_align(base_embed, other_embed, post_normalize=True):
    ''' Hiyam: This was the original code -- does not work for Gensim'''
    in_base_embed, in_other_embed = intersection_align(base_embed, other_embed, post_normalize=False)
    base_vecs = in_base_embed.m
    other_vecs = in_other_embed.m
    m = other_vecs.T.dot(base_vecs)
    u, _, v = np.linalg.svd(m) 
    ortho = u.dot(v)
    return Embedding((other_embed.m).dot(ortho), other_embed.iw, normalize = post_normalize)


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    base_embed.init_sims()
    other_embed.init_sims()

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # get the embedding matrices
    base_vecs = in_base_embed.wv.syn0norm
    other_vecs = in_other_embed.wv.syn0norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.syn0norm = other_embed.wv.syn0 = (other_embed.wv.syn0norm).dot(ortho)
    return other_embed


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    if m1.wv.syn0norm is None:
        m1.init_sims()
    if m2.wv.syn0norm is None:
        m2.init_sims()

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count, reverse=True)

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = m.wv.syn0norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.syn0norm = m.wv.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.wv.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = Vocab(index=new_index, count=old_vocab_obj.count)
        m.wv.vocab = new_vocab
        print(len(m.wv.index2word), len(m.wv.vocab))

    return (m1, m2)


def procrustes_align(base_embed, other_embed):
    """ 
        Align other embedding to base embeddings via Procrustes.
        Returns best distance-preserving aligned version of other_embed
        NOTE: Assumes indices are aligned
    """
    basevecs = base_embed.m - base_embed.m.mean(0)
    othervecs = other_embed.m - other_embed.m.mean(0)
    m = othervecs.T.dot(basevecs)
    u, _, v = np.linalg.svd(m) 
    ortho = u.dot(v)
    fixedvecs = othervecs.dot(ortho)
    return Embedding(fixedvecs, other_embed.iw)

def linear_align(base_embed, other_embed):
    """
        Align other embedding to base embedding using best linear transform.
        NOTE: Assumes indices are aligned
    """
    basevecs = base_embed.m
    othervecs = other_embed.m
    fixedvecs = othervecs.dot(np.linalg.pinv(othervecs)).dot(basevecs)
    return Embedding(fixedvecs, other_embed.iw)



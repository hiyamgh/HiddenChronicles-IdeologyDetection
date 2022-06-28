import fasttext
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import euclidean

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


# path = 'D:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/'
# years = [str(y) for y in range(1982, 2010)]
# for y in years:

print('year = {}'.format(1982))
# model = fasttext.load_model('D:/1982.bin')
model = fasttext.load_model('D:/fasttext_embeddings/ngrams4-size300-window5-mincount100-negative15-lr0.001/ngrams4-size300-window5-mincount100-negative15-lr0.001/1982.bin')
w1 = "فلسطين"
w2 = "مرحبا"
print('cosine similarity between {} and {} = {}'.format(w1, w2, cosine_similarity(model.get_word_vector(w1), model.get_word_vector(w2))))
print('euclidean distance between {} and {} = {}'.format(w1, w2, euclidean(model.get_word_vector(w1), model.get_word_vector(w2))))
sub1 = model.get_subwords("فلسطين")
sub2 = model.get_subwords("مرحبا")
print('subwords of {}: '.format(w1))
for sub in sub1:
    print(sub)

print('subwords of {}: '.format(w2))
for sub in sub2:
    print(sub)

results = model.get_nearest_neighbors("فلسطين", 100)
for r in results:
    print(r)
    # results = model.get_nearest_neighbors('فلسطين', 1000)
    # for r in results:
    #     print(r)
    # results = model.get_analogies("بيروت", "لبنان", "فرنسا")
    # for r in results:
    #     print(r)
    # print('==============================================================')
from gensim.models import Word2Vec

model = Word2Vec.load('E:/newspapers/word2vec/nahar/embeddings/word2vec_1933')
most_similar = model.wv.most_similar(positive=['الفلسطينيه'], topn=10)

# for w in most_similar:
#     print(w)
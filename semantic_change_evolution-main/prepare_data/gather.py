import os
import fasttext
import pickle

years = list(range(2000, 2011))
all_vocab = []
for y in years:
    model = fasttext.load_model('E:/fasttext_embeddings/ngrams4-size100-window3-mincount10-negative5-lr0.001/{}.bin'.format(y))
    all_vocab.append(model.words)

common_vocab = list(set.intersection(*map(set, all_vocab)))
# save common vocab
with open('parrot.pkl', 'wb') as f:
    pickle.dump(common_vocab, f)



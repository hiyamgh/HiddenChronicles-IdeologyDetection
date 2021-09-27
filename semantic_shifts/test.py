import fasttext

model1 = fasttext.load_model('../Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin')
model2 = fasttext.load_model('../Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin')

w = 'اسرائيل'
neighs1 = set(out[1] for out in model1.get_nearest_neighbors(w, 1000))
neighs2 = set(out[1] for out in model2.get_nearest_neighbors(w, 1000))

common = []
for nn1 in neighs1:
    for nn2 in neighs2:
        maxlen = max(len(nn1), len(nn2))
        # if nn1 in nn2 or nn2 in nn1:
        if maxlen - len(''.join(set(nn1).intersection(nn2))) <= 2:
           common.append((nn1, nn2))
           print((nn1, nn2))

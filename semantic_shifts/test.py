import fasttext
from nltk import everygrams
from nltk import ngrams
#
# w1 = 'تقوماسرائيل'
# w2 = 'توقفاسرائيل'


# lst = list(everygrams(w1, 1, 3))
# for e in lst:
# #     print(e)
# N = len(w1)
# grams = [w1[i:i+N] for i in range(len(w1)-N+1)]
# for g in grams:
#     print(g)
#


# ('', 'تقوماسرائيل')
# ('ت', 'قوماسرائيل')
# ('تق', 'وماسرائيل')
# ('تقو', 'ماسرائيل')
# ('تقوم', 'اسرائيل')
# ('تقوما', 'سرائيل')
# ('تقوماس', 'رائيل')
# ('تقوماسر', 'ائيل')
# ('تقوماسرا', 'ئيل')
# ('تقوماسرائ', 'يل')
# ('تقوماسرائي', 'ل')
# -----------------------------
# ('', 'توقفاسرائيل')
# ('ت', 'وقفاسرائيل')
# ('تو', 'قفاسرائيل')
# ('توق', 'فاسرائيل')
# ('توقف', 'اسرائيل')
# ('توقفا', 'سرائيل')
# ('توقفاس', 'رائيل')
# ('توقفاسر', 'ائيل')
# ('توقفاسرا', 'ئيل')
# ('توقفاسرائ', 'يل')
# ('توقفاسرائي', 'ل')

# for i in range(len(w1)):
#     print((w1[0:i], w1[i:]))
#
# print('-----------------------------')
#
# for i in range(len(w2)):
#     print((w2[0:i], w2[i:]))
#
# print((len(w1) - len('اسرائيل'))/ max(len('اسرائيل'), len(w1)))

model1 = fasttext.load_model('../Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin')
model2 = fasttext.load_model('../Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/2006.bin')
#
w = 'اسرائيل'
neighs1 = set(out[1] for out in model1.get_nearest_neighbors(w, 1000))
neighs2 = set(out[1] for out in model2.get_nearest_neighbors(w, 1000))
#
common = set()
for nn1 in neighs1:
    for nn2 in neighs2:
        maxlen = max(len(nn1), len(nn2))
        # if there is a huge intersection between the characheters of both words
        if maxlen - len(''.join(set(nn1).intersection(nn2))) <= 2:
            ngrams_nn1, ngrams_nn2 = [], []
            for i in range(len(nn1)):
                ngrams_nn1.append(nn1[0:i])
                ngrams_nn1.append(nn1[i:])
            for i in range(len(nn2)):
                ngrams_nn2.append(nn2[0:i])
                ngrams_nn2.append(nn2[i:])
            # get the intersection
            cmn = set(ngrams_nn1).intersection(set(ngrams_nn2))
            # sort by decreasing order of length
            cmn_sorted = sorted(list(cmn), key=lambda x: (-len(x), x))
            # get original word for comparison
            original = nn1 if len(nn1) >= len(nn2) else nn2
            # if there exist a word in the intersection that is less than the original word by 40% max, then add it
            diffs = {}
            for cw in cmn:
                if abs((len(original) - len(cw))) / len(original) <= 0.4:
                    # common.add(nn1 if len(nn1) >= len(nn2) else nn2)
                    common.add(cw)
                    print('{} is subset of {}, added {}'.format(cw, original, (nn1, nn2)))
                    break

print('============================================================================================')
print(len(common))
print(common)
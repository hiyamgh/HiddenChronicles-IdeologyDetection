import pickle, os
import fasttext
import pandas as pd
import io

embedding1 = '../Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/'
embedding2 = '../Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/'

years = list(range(2000, 2005))
for year in years:

    # df = pd.DataFrame(columns=['word', 'score', 'nearest neighs'])

    model1 = fasttext.load_model(os.path.join(embedding1, '{}.bin'.format(year)))
    model2 = fasttext.load_model(os.path.join(embedding2, '{}.bin'.format(year)))

    with open(os.path.join('nn_scores_nahar_{}_assafir_{}.pkl'.format(year, year)), 'rb') as f:
        w2nnscore = pickle.load(f)
        for tp in w2nnscore[:100]:

            w = tp[1]
            score = tp[0]

            strofnns = ''
            nn1 = model1.get_nearest_neighbors(w, 100)
            for i, nnsim in enumerate(nn1):
                if i%10 != 0:
                    strofnns += '{},'.format(nnsim[1])
                else:
                    strofnns += '\n{},'.format(nnsim[1])
            strofnns+= '\n=============================='
            nn2 = model2.get_nearest_neighbors(w, 100)
            for i, nnsim in enumerate(nn2):
                if i%10 != 0:
                    strofnns += '{},'.format(nnsim[1])
                else:
                    strofnns += '\n{},'.format(nnsim[1])

            print(strofnns)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # df = df.append({
            #     'word': w,
            #     'score': score,
            #     'nearest neighs': strofnns.encode('utf-8')
            # }, ignore_index=True)

            # with io.open('nn_scores_nahar_{}_assafir_{}.csv'.format(year, year), 'w', encoding='utf-8-sig') as f:
            #     s = u','.join([w, str(score), strofnns]) + u'\n'
            #     f.write(s)
            # f.close()

        # df.to_csv('nn_scores_nahar_{}_assafir_{}.csv'.format(year, year))

        # path = 'E:/'
        # years = list(range(2000, 2005))
        # for year in years:
        #     with open(os.path.join(path, 'nn_scores_nahar_{}_assafir_{}.pkl'.format(year, year)), 'rb') as f:
        #         res = pickle.load(f)
        #         for e in res:
        #             # e[0] = nb of intersection of nn of the word between embed_0 and embed_1
        #             # e[1] = the word
        #             print()
        #         # print('year: {}: num={}'.format(year, len(res)))
        #
        # w = 'الاسرائيلي'


        # for l in l1[50:]:
            #     print(l[1])
            # print('=======================================================')
            # l2 = model2.get_nearest_neighbors(w, 100)
            # for l in l2[50:]:
            #     print(l[1])

# with open('nn_scores_nahar_{}_assafir_{}.pkl'.format(year, year), 'rb') as f:
#     res = pickle.load(f)
# for e in res:
#     # e[0] = nb of intersection of nn of the word between embed_0 and embed_1
#     # e[1] = the word
#      print()
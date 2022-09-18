# from arabert.preprocess import ArabertPreprocessor
#
# model_name = "aubmindlab/bert-base-arabertv2"
# arabert_prep = ArabertPreprocessor(model_name=model_name)
#
# text = "ولن نبالغ إذا قلنا: إن 'هاتف' أو 'كمبيوتر المكتب' في زمننا هذا ضروري"
# print(arabert_prep.preprocess(text))
#

import pickle
with open('embeddings/nahar-1982-1986.pickle', 'rb') as handle:
    b = pickle.load(handle)
print()

words = ['مقاومه', 'اسرائيل', 'فلسطين']
for k in b:
    for w in words:
        if (w == k) or (w in k):
            print(k)


# import os
#
# import fasttext
# import torch
# import numpy as np
# import time
# import pandas as pd
# # ft = fasttext.load_model('../cc.ar.300.bin')
# # t1 = time.time()
# # print(ft.get_input_matrix().shape)
# # t2 = time.time()
# # print('time elapsed: {:.2f} mins'.format((t2-t1)/60))
# # # word_vectors = torch.from_numpy(ft.get_input_matrix())
# # # classifier_fc = torch.from_numpy(ft.get_output_matrix())
#
# # df = pd.read_excel('C:/Users/96171/Downloads/df_test_1982.xlsx')
# # print(df['Prediction'].value_counts())
#
# # df = pd.read_excel('C:/Users/96171/Downloads/df_test_1986.xlsx')
# # print((df['Prediction'].value_counts()/len(df)) * 100)
#
# # df = pd.read_excel('C:/Users/96171/Downloads/df_test_1982.xlsx')
# # print((df['Prediction'].value_counts()/len(df)) * 100)
#
# # path = 'bert_predictions/nahar/'
# # for file in ['df_test_{}_Mukawama.xlsx'.format(y) for y in [1982, 1985, 1986, 2005, 2006, 2007]]:
# #     df = pd.read_excel(os.path.join(path, file))
# #     unique_labels = list(set(df['Prediction']))
# #     print('{}: {}'.format(file, unique_labels))
#
# for file in ['df_test_{}_Mukawama.xlsx'.format(y) for y in [1982, 1985, 1986, 2005, 2006, 2007]]:
#     # df = pd.read_excel('C:/Users/96171/Downloads/{}'.format(file))
#     df = pd.read_excel('C:/Users/96171/Downloads/predictions-dp/{}'.format(file))
#     print(list(set(df['Prediction'])))
#
#     print(df['Prediction'].value_counts())
#     print((df['Prediction'].value_counts()/len(df))*100)
#     print('==========================================================')
#
# # df_webis16_train = pd.read_excel('input/corpus-webis-editorials-16/df_train.xlsx')
# # df_webis16_dev = pd.read_excel('input/corpus-webis-editorials-16/df_dev.xlsx')
# # df_webis16_test = pd.read_excel('input/corpus-webis-editorials-16/df_test.xlsx')
# #
# # print('df_train: {}'.format(df_webis16_train.shape))
# # print('df_dev: {}'.format(df_webis16_dev.shape))
# # print('df_train: {}'.format(df_webis16_test.shape))
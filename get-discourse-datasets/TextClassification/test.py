import fasttext
import torch
import numpy as np
import time

ft = fasttext.load_model('../cc.ar.300.bin')
t1 = time.time()
print(ft.get_input_matrix().shape)
t2 = time.time()
print('time elapsed: {:.2f} mins'.format((t2-t1)/60))
# word_vectors = torch.from_numpy(ft.get_input_matrix())
# classifier_fc = torch.from_numpy(ft.get_output_matrix())


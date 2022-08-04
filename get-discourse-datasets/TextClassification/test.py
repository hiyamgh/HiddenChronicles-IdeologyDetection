import fasttext
import torch
import numpy as np

ft = fasttext.load_model('../cc.ar.300.bin')
word_vectors = torch.from_numpy(ft.get_input_matrix())
classifier_fc = torch.from_numpy(ft.get_output_matrix())


import os

# with open('E:/fasttext_embeddings/2009.txt', 'r', encoding='utf-8') as f:
#     content = f.readlines(100000)
# f.close()
#
# with open('E:/fasttext_embeddings/2009-split.txt', 'w', encoding='utf-8') as f:
#     for line in content:
#         f.write(line)
#     f.close()

with open('E:/fasttext_embeddings/2008.txt', 'r', encoding='utf-8') as f:
    content = f.readlines(100000)
f.close()

with open('E:/fasttext_embeddings/2008-split.txt', 'w', encoding='utf-8') as f:
    for line in content:
        f.write(line)
    f.close()

# from transformers import BertTokenizer, BertForMaskedLM
# import torch
#
# tokenizer = BertTokenizer.from_pretrained('aubmindlab/bert-base-arabertv1')
# model = BertForMaskedLM.from_pretrained('aubmindlab/bert-base-arabertv2')

# this worked
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# from transformers import RobertaTokenizer, RobertaForMaskedLM
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaForMaskedLM.from_pretrained('roberta-base')

# tokenizer = BertTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')
# model = BertForMaskedLM.from_pretrained('aubmindlab/bert-base-arabertv2', from_tf=True)
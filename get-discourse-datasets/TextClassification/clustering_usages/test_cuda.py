
import torch
from transformers import BertModel, BertTokenizer
from arabert import ArabertPreprocessor

pretrained_weights = "aubmindlab/bert-base-arabertv2"
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
model = BertModel.from_pretrained(pretrained_weights)
arabert_prep = ArabertPreprocessor(model_name=pretrained_weights)
if torch.cuda.is_available():
    model.to('cuda')
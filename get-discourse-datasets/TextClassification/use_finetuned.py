from transformers import BertForSequenceClassification, BertConfig
import torch

# model = BertForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv2")
# model.load_state_dict(torch.load("bert_output/pytorch_model.bin"))

config = BertConfig.from_pretrained("aubmindlab/bert-base-arabertv2", num_labels=8)
model = BertForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv2", config=config)
model.load_state_dict(torch.load("bert_output/pytorch_model.bin"))


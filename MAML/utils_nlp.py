# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix, accuracy_score

from copy import deepcopy
from arabert.preprocess import ArabertPreprocessor
import pandas as pd
import matplotlib.pyplot as plt
import json

from transformers import (
    WEIGHTS_NAME,
    CONFIG_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    AutoTokenizer,
    BertForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

# logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(format='%(message)s',  # "format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


class DiscourseProcessor:

    def __init__(self):
        self.text_col = "Sentence"
        self.label_col = "Label"
        self.train_examples, self.dev_examples, self.test_examples = [], [], []

    def _read_dataset(self, df_path):
        if '.csv' in df_path or '.xlsx' in df_path:
            df = pd.read_csv(df_path) if '.csv' in df_path else pd.read_excel(df_path)

            # drop any NaN values
            print('before dropping nans: df.shape: {}'.format(df.shape))
            df = df.dropna()
            print('after dropping nans: df.shape: {}'.format(df.shape))

            sentences = []
            for i, row in df.iterrows():
                sentence = str(row[self.text_col])
                if sentence.strip().isdigit():
                    continue

                label = str(row[self.label_col])
                sentences.append([sentence, label])

            return sentences

    def get_train_examples(self, df_path):
        print('reading the training dataset from {} ...'.format(df_path))
        return self._create_examples(self._read_dataset(df_path=df_path), "train")

    def get_dev_examples(self, df_path):
        print('reading the validation dataset from {} ...'.format(df_path))
        return self._create_examples(self._read_dataset(df_path=df_path), "dev")

    def get_test_examples(self, df_path):
        print('reading the testing dataset from {} ...'.format(df_path))
        return self._create_examples(self._read_dataset(df_path=df_path), "test")

    def _create_examples(self, data_tuples, set_type):
        """ Creates examples for the train and test sets """
        exampels = []
        for i, data_tuple in enumerate(data_tuples):
            guid = "%s-%s" % (set_type, i)
            sentence, label = data_tuple

            exampels.append(self._get_input_example(guid=guid, sentence=sentence, label=label))

        return exampels

    def _get_input_example(self, guid, sentence, label):
        return InputExample(guid=guid, text_a=sentence, label=label)

    def get_labels(self):
        return ["Distant_Evaluation", "Distant_Expectations_Consequences", "Main_Consequence", "Cause_General", "Cause_Specific", "Main",
                "Distant_Historical", "Distant_Anecdotal"]

    def load_all_data(self, df_train_path, df_dev_path, df_test_path):
        self.train_examples = self.get_train_examples(df_path=df_train_path)
        # self.dev_examples = self.get_dev_examples(df_path=df_dev_path)
        self.test_examples = self.get_test_examples(df_path=df_test_path)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


tasks_num_labels = {

    "dp": 8,
}

processors = {
    "dp": DiscourseProcessor,
}

output_modes = {
    "dp": "classification",
}



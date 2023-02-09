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

    def __init__(self, labels, preprocess_arabert=False, model_name=None):
        self.labels = labels
        self.train_examples, self.dev_examples, self.test_examples = [], [], []
        self.preprocess_arabert = preprocess_arabert
        self.model_name = model_name

    def _read_dataset(self, df_path, df_id):
        if "corp" in df_id:
            text_col = "Sentence"
            label_col = "label"
        else:
            if "PTC" in df_id:
                text_col = "Sentence"
                label_col = "Label"
            elif "ARG" in df_id:
                text_col = "Sentence"
                label_col = "Label"
            elif "VDC" in df_id:
                text_col = "Sentence"
                label_col = "Label"
            else:
                # VDS
                text_col = "Sentence"
                label_col = "Speech_label"

        df = pd.read_csv(df_path) if '.csv' in df_path else pd.read_excel(df_path)

        # drop any NaN values
        print('before dropping nans: df.shape: {}'.format(df.shape))
        df = df.dropna()
        print('after dropping nans: df.shape: {}'.format(df.shape))

        sentences = []

        if self.preprocess_arabert and self.model_name is not None:
            print('preprocessing text through ArabertPreprocessor(model_name={})'.format(self.model_name))
            arabert_prep = ArabertPreprocessor(model_name=self.model_name)
            for i, row in df.iterrows():
                sentence = str(row[text_col])
                if sentence.strip().isdigit():
                    continue
                sentence = arabert_prep.preprocess(sentence)
                label = str(row[label_col])
                sentences.append([sentence, label])
        else:
            for i, row in df.iterrows():
                sentence = str(row[text_col])
                if sentence.strip().isdigit():
                    continue
                label = str(row[label_col])
                sentences.append([sentence, label])

        return sentences

    def get_examples(self, df_path, df_id):
        print("reading the dataset from {} ...".format(df_path))
        return self._create_examples(self._read_dataset(df_path=df_path, df_id=df_id), "train")

    def _create_examples(self, data_tuples, set_type):
        """ Creates examples for the train and test sets """
        examples = []
        for i, data_tuple in enumerate(data_tuples):
            guid = "%s-%s" % (set_type, i)
            sentence, label = data_tuple

            examples.append(self._get_input_example(guid=guid, sentence=sentence, label=label))

        return examples

    def _get_input_example(self, guid, sentence, label):
        return InputExample(guid=guid, text_a=sentence, label=label)

    def get_labels(self, ptc=False):
        labels = self.labels.split(";") if ptc else self.labels.split(",")
        labels = [l.strip() for l in labels if l.strip() != ""]
        return labels


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


def load_cache_examples(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset


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


processors = {
    "dp": DiscourseProcessor,
}

output_modes = {
    "dp": "classification",
}

# VDC_lang --> Discourse Profiling
# PTC_lang --> Propaganda
# ARG_lang --> Argumentation

# Our corpus - (P)alestinian (R)esistance (S)outh Lebanon (PRST) 307 sentences
# corpus_PRST_lang_VDS
# corpus_PRST_lang_VDC
# corpus_PRST_lang_PTC
# corpus_PRST_lang_ARG

# Our corpus - (S)abra (S)hatila (M)assacre (SSM) 181 sentences
# corpus_SSM_lang_VDS
# corpus_SSM_lang_VDC
# corpus_SSM_lang_PTC
# corpus_SSM_lang_ARG

# en only included when we have our corpus (we want a translation to English)
# in other scenarios, we take ar as the target language
codes2names = {
    'fr': 'french',
    'es': 'spanish',
    'de': 'german',
    'el': 'greek',
    'bg': 'bulgarian',
    'ru': 'russian',
    'tr': 'turkish',
    'ar': 'arabic',
    'en': 'english',
    'vi': 'vietnamese',
    'th': 'thai',
    'zh-cn': 'chinese (simplified)',
    'hi': 'hindi',
    'sw': 'swahili',
    'ur': 'urdu'
}

datasets = {
    # Discourse Profiling (Contexts)
    "VDC_de": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_de.xlsx",
    "VDC_th": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_th.xlsx",
    "VDC_zh-cn": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_zh-cn.xlsx",
    "VDC_ru": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_ru.xlsx",
    "VDC_hi": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_hi.xlsx",
    "VDC_bg": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_bg.xlsx",
    "VDC_el": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_el.xlsx",
    "VDC_sw": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_sw.xlsx",
    "VDC_es": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_es.xlsx",
    "VDC_tr": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_tr.xlsx",
    "VDC_fr": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_fr.xlsx",
    "VDC_vi": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_vi.xlsx",
    "VDC_ur": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_ur.xlsx",
    "VDC_ar": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_ar.xlsx",
    "VDC_en": "../translate_corpora/Discourse_Profiling/df.xlsx",

    # Discourse Profiling (Speeches)
    "VDS_de": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_de.xlsx",
    "VDS_th": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_th.xlsx",
    "VDS_zh-cn": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_zh-cn.xlsx",
    "VDS_ru": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_ru.xlsx",
    "VDS_hi": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_hi.xlsx",
    "VDS_bg": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_bg.xlsx",
    "VDS_el": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_el.xlsx",
    "VDS_sw": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_sw.xlsx",
    "VDS_es": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_es.xlsx",
    "VDS_tr": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_tr.xlsx",
    "VDS_fr": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_fr.xlsx",
    "VDS_vi": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_vi.xlsx",
    "VDS_ur": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_ur.xlsx",
    "VDS_ar": "../translate_corpora/Discourse_Profiling/translations_joined_cleaned/df_ar.xlsx",
    "VDS_en": "../translate_corpora/Discourse_Profiling/df.xlsx",


    # Propaganda
    "PTC_de": "../translate_corpora/ptc-corpus/translations_joined/df_de.xlsx",
    "PTC_th": "../translate_corpora/ptc-corpus/translations_joined/df_th.xlsx",
    "PTC_zh-cn": "../translate_corpora/ptc-corpus/translations_joined/df_zh-cn.xlsx",
    "PTC_ru": "../translate_corpora/ptc-corpus/translations_joined/df_ru.xlsx",
    "PTC_hi": "../translate_corpora/ptc-corpus/translations_joined/df_hi.xlsx",
    "PTC_bg": "../translate_corpora/ptc-corpus/translations_joined/df_bg.xlsx",
    "PTC_el": "../translate_corpora/ptc-corpus/translations_joined/df_el.xlsx",
    "PTC_sw": "../translate_corpora/ptc-corpus/translations_joined/df_sw.xlsx",
    "PTC_es": "../translate_corpora/ptc-corpus/translations_joined/df_es.xlsx",
    "PTC_tr": "../translate_corpora/ptc-corpus/translations_joined/df_tr.xlsx",
    "PTC_fr": "../translate_corpora/ptc-corpus/translations_joined/df_fr.xlsx",
    "PTC_vi": "../translate_corpora/ptc-corpus/translations_joined/df_vi.xlsx",
    "PTC_ur": "../translate_corpora/ptc-corpus/translations_joined/df_ur.xlsx",
    "PTC_ar": "../translate_corpora/ptc-corpus/translations_joined/df_ar.xlsx",
    "PTC_en": "../translate_corpora/ptc-corpus/annotations/df_multiclass.xlsx",


    # Argumentation
    "ARG_th": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_th.xlsx",
    "ARG_hi": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_hi.xlsx",
    "ARG_fr": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_fr.xlsx",
    "ARG_zh-cn": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_zh-cn.xlsx",
    "ARG_sw": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_sw.xlsx",
    "ARG_bg": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_bg.xlsx",
    "ARG_vi": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_vi.xlsx",
    "ARG_ru": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_ru.xlsx",
    "ARG_ar": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_ar.xlsx",
    "ARG_tr": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_tr.xlsx",
    "ARG_ur": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_ur.xlsx",
    "ARG_el": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_el.xlsx",
    "ARG_de": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_de.xlsx",
    "ARG_es": "../translate_corpora/corpus-webis-editorials-16/translationsv2_cleaned/sentences_annotations_es.xlsx",
    "ARG_en": "sentences_annotations.xlsx",

    # Our Corpus
    "corp_PRST_sw_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_sw.xlsx",
    "corp_PRST_th_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_th.xlsx",
    "corp_PRST_ru_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_ru.xlsx",
    "corp_PRST_bg_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_bg.xlsx",
    "corp_PRST_ur_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_ur.xlsx",
    "corp_PRST_ar_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_ar.xlsx",
    "corp_PRST_tr_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_tr.xlsx",
    "corp_PRST_en_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_en.xlsx",
    "corp_PRST_de_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_de.xlsx",
    "corp_PRST_zh-cn_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_zh-cn.xlsx",
    "corp_PRST_es_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_es.xlsx",
    "corp_PRST_vi_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_vi.xlsx",
    "corp_PRST_hi_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_hi.xlsx",
    "corp_PRST_fr_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_fr.xlsx",
    "corp_PRST_el_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/our_corpus_el.xlsx",
    "corp_SSM_bg_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_bg.xlsx",
    "corp_SSM_ar_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_ar.xlsx",
    "corp_SSM_ru_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_ru.xlsx",
    "corp_SSM_es_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_es.xlsx",
    "corp_SSM_de_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_de.xlsx",
    "corp_SSM_sw_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_sw.xlsx",
    "corp_SSM_el_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_el.xlsx",
    "corp_SSM_ur_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_ur.xlsx",
    "corp_SSM_zh-cn_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_zh-cn.xlsx",
    "corp_SSM_hi_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_hi.xlsx",
    "corp_SSM_en_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_en.xlsx",
    "corp_SSM_th_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_th.xlsx",
    "corp_SSM_tr_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_tr.xlsx",
    "corp_SSM_fr_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_fr.xlsx",
    "corp_SSM_vi_VDS": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/our_corpus_vi.xlsx",

    "corp_PRST_sw_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_sw.xlsx",
    "corp_PRST_th_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_th.xlsx",
    "corp_PRST_ru_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_ru.xlsx",
    "corp_PRST_bg_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_bg.xlsx",
    "corp_PRST_ur_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_ur.xlsx",
    "corp_PRST_ar_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_ar.xlsx",
    "corp_PRST_tr_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_tr.xlsx",
    "corp_PRST_en_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_en.xlsx",
    "corp_PRST_de_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_de.xlsx",
    "corp_PRST_zh-cn_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_zh-cn.xlsx",
    "corp_PRST_es_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_es.xlsx",
    "corp_PRST_vi_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_vi.xlsx",
    "corp_PRST_hi_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_hi.xlsx",
    "corp_PRST_fr_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_fr.xlsx",
    "corp_PRST_el_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/our_corpus_el.xlsx",
    "corp_SSM_bg_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_bg.xlsx",
    "corp_SSM_ar_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_ar.xlsx",
    "corp_SSM_ru_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_ru.xlsx",
    "corp_SSM_es_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_es.xlsx",
    "corp_SSM_de_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_de.xlsx",
    "corp_SSM_sw_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_sw.xlsx",
    "corp_SSM_el_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_el.xlsx",
    "corp_SSM_ur_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_ur.xlsx",
    "corp_SSM_zh-cn_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_zh-cn.xlsx",
    "corp_SSM_hi_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_hi.xlsx",
    "corp_SSM_en_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_en.xlsx",
    "corp_SSM_th_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_th.xlsx",
    "corp_SSM_tr_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_tr.xlsx",
    "corp_SSM_fr_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_fr.xlsx",
    "corp_SSM_vi_VDC": "../translate_corpora/annotations_doccano/translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/our_corpus_vi.xlsx",


    "corp_PRST_sw_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_sw.xlsx",
    "corp_PRST_th_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_th.xlsx",
    "corp_PRST_ru_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_ru.xlsx",
    "corp_PRST_bg_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_bg.xlsx",
    "corp_PRST_ur_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_ur.xlsx",
    "corp_PRST_ar_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_ar.xlsx",
    "corp_PRST_tr_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_tr.xlsx",
    "corp_PRST_en_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_en.xlsx",
    "corp_PRST_de_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_de.xlsx",
    "corp_PRST_zh-cn_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_zh-cn.xlsx",
    "corp_PRST_es_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_es.xlsx",
    "corp_PRST_vi_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_vi.xlsx",
    "corp_PRST_hi_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_hi.xlsx",
    "corp_PRST_fr_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_fr.xlsx",
    "corp_PRST_el_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Palestinian Resistance South Lebanon/our_corpus_el.xlsx",
    "corp_SSM_sw_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_sw.xlsx",
    "corp_SSM_th_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_th.xlsx",
    "corp_SSM_ru_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_ru.xlsx",
    "corp_SSM_bg_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_bg.xlsx",
    "corp_SSM_ur_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_ur.xlsx",
    "corp_SSM_ar_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_ar.xlsx",
    "corp_SSM_tr_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_tr.xlsx",
    "corp_SSM_en_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_en.xlsx",
    "corp_SSM_de_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_de.xlsx",
    "corp_SSM_zh-cn_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_zh-cn.xlsx",
    "corp_SSM_es_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_es.xlsx",
    "corp_SSM_vi_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_vi.xlsx",
    "corp_SSM_hi_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_hi.xlsx",
    "corp_SSM_fr_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_fr.xlsx",
    "corp_SSM_el_PTC": "../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_el.xlsx",

    "corp_PRST_bg_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_bg.xlsx",
    "corp_PRST_ar_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_ar.xlsx",
    "corp_PRST_ru_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_ru.xlsx",
    "corp_PRST_es_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_es.xlsx",
    "corp_PRST_de_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_de.xlsx",
    "corp_PRST_sw_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_sw.xlsx",
    "corp_PRST_el_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_el.xlsx",
    "corp_PRST_ur_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_ur.xlsx",
    "corp_PRST_zh-cn_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_zh-cn.xlsx",
    "corp_PRST_hi_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_hi.xlsx",
    "corp_PRST_en_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_en.xlsx",
    "corp_PRST_th_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_th.xlsx",
    "corp_PRST_tr_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_tr.xlsx",
    "corp_PRST_fr_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_fr.xlsx",
    "corp_PRST_vi_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Palestinian Resistance South Lebanon/our_corpus_vi.xlsx",
    "corp_SSM_sw_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_sw.xlsx",
    "corp_SSM_th_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_th.xlsx",
    "corp_SSM_ru_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_ru.xlsx",
    "corp_SSM_bg_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_bg.xlsx",
    "corp_SSM_ur_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_ur.xlsx",
    "corp_SSM_ar_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_ar.xlsx",
    "corp_SSM_tr_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_tr.xlsx",
    "corp_SSM_en_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_en.xlsx",
    "corp_SSM_de_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_de.xlsx",
    "corp_SSM_zh-cn_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_zh-cn.xlsx",
    "corp_SSM_es_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_es.xlsx",
    "corp_SSM_vi_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_vi.xlsx",
    "corp_SSM_hi_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_hi.xlsx",
    "corp_SSM_fr_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_fr.xlsx",
    "corp_SSM_el_ARG": "../translate_corpora/annotations_doccano/translationsv2/argumentation/Sabra and Shatila Massacre/our_corpus_el.xlsx",
}
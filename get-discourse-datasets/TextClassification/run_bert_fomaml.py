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

logging.basicConfig(format='%(message)s', #"format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class Webis16Processor:

    def __init__(self, model_name):
        self.labels = None
        self.model_name = model_name
        self.text_col = "Sentence"
        self.label_col = "Label"
        self.train_examples, self.dev_examples, self.test_examples = [], [], []
        self.train_labels, self.dev_labels, self.test_labels = [], [], []

    def _read_dataset(self, df_path):
        arabert_prep = ArabertPreprocessor(model_name=self.model_name)
        if '.csv' in df_path or '.xlsx' in df_path:
            df = pd.read_csv(df_path) if '.csv' in df_path else pd.read_excel(df_path)
            arabert_prep = ArabertPreprocessor(model_name=self.model_name)

            sentences = []
            for i, row in df.iterrows():
                sentence = str(row[self.text_col])
                if sentence.strip().isdigit():
                    continue
                # according to https://github.com/aub-mind/arabert#preprocessing
                # It is recommended to apply our preprocessing function before training/testing on any dataset
                sentence = arabert_prep.preprocess(sentence)
                label = str(row[self.label_col])

                sentences.append([sentence, label])

            return sentences
        else:
            list_of_paths = df_path.split(';')
            sentences = []
            for df_path in list_of_paths:
                with open(df_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                json_file.close()
                for k in data:
                    sentence_full = ''
                    label = ''
                    if data[k]['label'].strip() != "":
                        for sentence in data[k]:
                            if data[k][sentence].strip().isdigit():
                                continue
                            if sentence not in ['keywords', 'year', 'label']:
                                sentence_full += data[k][sentence] + ' '
                            # if sentence == 'label':
                                # label_list = [e for e in data[k][sentence].split(";") if e.strip() != ""]
                                # all_labels = self.get_labels()
                                # label = [1.0 if l in label_list else 0.0 for l in all_labels]

                        sentence_full = arabert_prep.preprocess(sentence_full)
                        sentences.append([sentence_full, data[k]['label'].strip()])
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
        labels = []
        for i, data_tuple in enumerate(data_tuples):
            guid = "%s-%s" % (set_type, i)
            sentence, label = data_tuple

            exampels.append(self._get_input_example(guid=guid, sentence=sentence, label=label))
            labels.append(label)

        return exampels, labels

    def _get_input_example(self, guid, sentence, label):
        return InputExample(text_a=sentence, label=label, guid=guid)

    def get_labels(self):
        return ["statistics", "anecdote", "common-ground", "assumption", "testimony"]

    def load_all_data(self, df_train_path, df_dev_path, df_test_path):
        self.train_examples, self.train_labels = self.get_train_examples(df_path=df_train_path)
        self.dev_examples, self.dev_labels = self.get_dev_examples(df_path=df_dev_path)
        self.test_examples, self.test_labels = self.get_test_examples(df_path=df_test_path)
    
    def get_test_tasks(self):
        return self.test_examples, self.test_labels

    def _sample_mini_dataset(self, examples, num_classes, num_shots):
        """
        Sample a few shot task from a dataset.

        :param examples: training/dev/test examples
        :param num_classes: number of classes
        :param num_shots: number of examples per class
        :return:
            An iterable of (input, label) pairs.
        """
        random.shuffle(examples)
        label_names = self.get_labels()
        for cls in range(num_classes):
            cls_indices = [i for i, example in enumerate(examples) if example.label == label_names[cls]] # get indices of all examples pertaining to the class cls
            selected = np.random.choice(cls_indices, num_shots, replace=False) # randomly select num_shots number of indices
            for i in selected:
                yield examples[i]


class DiscourseProfilingProcessor(Webis16Processor):
    def __init__(self, model_name):
        Webis16Processor.__init__(self, model_name)
        self.text_col = "Sentence_ar"
        self.label_col = "Label"

    def get_labels(self):
        return ["Cause_Specific", "Distant_Anecdotal", "Main_Consequence", "Distant_Expectations_Consequences", "Main",
                "Distant_Historical", "Cause_General", "Distant_Evaluation"]


class PTCProcessor(Webis16Processor):

    def __init__(self, model_name):
        Webis16Processor.__init__(self, model_name)
        self.text_col = "context_ar"
        self.label_col = "label"

    def get_labels(self):
        return ["Name_Calling,Labeling", "Whataboutism,Straw_Men,Red_Herring", "Loaded_Language", "Repetition",
                "Causal_Oversimplification", "Appeal_to_Authority", "Exaggeration,Minimisation", "Slogans", "Doubt",
                "Thought-terminating_Cliches", "Bandwagon,Reductio_ad_hitlerum", "Flag-Waving",
                "Appeal_to_fear-prejudice", "Black-and-White_Fallacy"]


class FAKESProcessor(Webis16Processor):

    def __init__(self, model_name):
        Webis16Processor.__init__(self, model_name)
        self.text_col = "article_content"
        self.label_col = "label"

    def _read_dataset(self, df_path):
        df = pd.read_csv(df_path) if '.csv' in df_path else pd.read_excel(df_path)

        sentences = []
        for i, row in df.iterrows():
            sentence = str(row[self.text_col])
            if sentence.strip().isdigit():
                continue
            label = str(row[self.label_col])
            sentences.append([sentence, label])

        return sentences

    def get_labels(self):
        return ["0", "1"]


def _mini_batches(samples, batch_size, num_batches, replacement):
    """
        Generate mini-batches from some data.
        :param samples: a mini-dataset containing num_shots*num_classes number of examples (per class)
        :param batch_size: size of a single batch
        :param num_batches: number of batches to generate
        :param replacement: boolean: whetehr to sample examples w/o replacement
        :return:
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

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
                              label_id=label_id))
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    # check if binary or multiclass classification
    # if len(list(set(labels))) == 2:
    #     accuracy = accuracy_score(labels, preds)
    #     f1 = f1_score(labels, preds)
    #     precision = precision_score(labels, preds)
    #     recall = recall_score(labels, preds)
    #     return {
    #         "acc": accuracy,
    #         "precision": precision,
    #         "recall": recall,
    #         "f1": f1,
    #         "acc_and_f1": (accuracy + f1) / 2,
    #     }
    # else:

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    precision = precision_score(y_true=labels, y_pred=preds, average="macro")
    recall = recall_score(y_true=labels, y_pred=preds, average="macro")
    return {
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc_and_f1": (accuracy + f1) / 2,
    }


def get_train_prob(reward_prob, K, epsilon):
    indices = np.argpartition(reward_prob, -K)[-K:]
    one_hot_prob = np.zeros((len(reward_prob)))
    for ind in indices:
        one_hot_prob[ind] = 1
    one_hot_prob = epsilon*np.ones(len(reward_prob))+one_hot_prob

    return one_hot_prob/np.sum(one_hot_prob)


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "ptc":
        return acc_and_f1(preds, labels)
    elif task_name == "webis":
        return acc_and_f1(preds, labels)
    elif task_name == "discourse_profiling":
        return acc_and_f1(preds, labels)
    elif task_name == "fakes":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


def accuracy(pred, label):
    '''
    pred: prediction result
    label: label with whatever size
    return: Accuracy[A single Value]
    '''
    return torch.mean((pred.view(-1)) == (label.view(-1)).type(torch.FloatTensor))


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        # default=None,
                        default="data/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="aubmindlab/bert-base-arabertv2", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        # default=None,
                        default="discourse_profiling",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        # default=None,
                        default="bert_output_discp/",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--is_init",
                        default=False,
                        type=bool,
                        help="whether initialize the model")
    parser.add_argument("--is_reptile",
                        default=False,
                        type=bool,
                        help="whether use reptile or fomaml method")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        # default=128,
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        # action='store_true',
                        action='store_false',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        # action='store_true',
                        action='store_false',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--inner_learning_rate",
                        default=2e-6,
                        type=float,
                        help="The inner learning rate for Adam")
    parser.add_argument("--outer_learning_rate",
                        default=1e-5,
                        type=float,
                        help="The meta learning rate for Adam, actual learning rate!")
    parser.add_argument("--FSL_learning_rate",
                        default=2e-5,
                        type=float,
                        help="The FSL learning rate for Adam!")
    parser.add_argument("--FSL_epochs",
                        default=1,
                        type=int,
                        help="The FSL learning epochs for training!")
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--inner_iters', type=int, default=20,
                        help='inner iterations')

    parser.add_argument('--inner_batch', type=int, default=5,
                        help='inner batch size')

    parser.add_argument('--num_shots', type=int, default=5,
                        help='number of examples per class')

    parser.add_argument('--meta_batch', type=int, default=1,
                        help='meta training batch size')

    parser.add_argument('--meta_iters', type=int, default=700,
                        help='meta-training iterations')

    parser.add_argument("--train_set",
                        default="input/Discourse_Profiling/df_train_cleaned.xlsx",
                        type=str,
                        help="path to the training dataset.")

    parser.add_argument("--dev_set",
                        default="input/Discourse_Profiling/df_dev_cleaned.xlsx",
                        type=str,
                        help="path to the training dataset.")

    parser.add_argument("--test_set",
                        default="sentences/labels_discourse_profiling/group_0_1982.json;sentences/labels_discourse_profiling/group_0_1984.json;sentences/labels_discourse_profiling/group_0_1985.json;sentences/labels_discourse_profiling/group_0_1986.json;sentences/labels_discourse_profiling/group_0_1987.json;sentences/labels_discourse_profiling/group_1_1982.json;sentences/labels_discourse_profiling/group_1_1983.json;sentences/labels_discourse_profiling/group_1_1984.json",
                        type=str,
                        help="path to the testing dataset.")


    args = parser.parse_args()

    processors = {
        "ptc": PTCProcessor,
        "discourse_profiling": DiscourseProfilingProcessor,
        "webis": Webis16Processor,
        "fakes": FAKESProcessor
    }

    output_modes = {
        "ptc": "classification",
        "discourse_profiling": "classification",
        "webis": "classification",
        "fakes": "classification"
    }

    train_path = args.train_set
    dev_path = args.dev_set
    test_path = args.test_set

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](args.bert_model)
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    print("read data!")
    # train_task_number, fsl_task_number = processor.calculate_task_num(args.data_dir)
    processor.load_all_data(df_train_path=train_path, df_dev_path=dev_path, df_test_path=test_path) # train, dev, and test examples are saved as lists of InputExamples
    print("load finished!")
    
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    proto_hidden = 100

    inner_iters = args.inner_iters # number of inner iterations
    meta_iters = args.meta_iters  # number of meta train iterations
    num_shots = args.num_shots # number of examples per class in the support set
    meta_batch = args.meta_batch  # meta_batch_size: number of tasks to sample **per meta iteration**
    N_class = num_labels # number of classes in each task
    inner_batch = args.inner_batch # ??
    Is_reptile = args.is_reptile
    if args.do_train :
        # train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = meta_iters*(num_shots)*N_class*meta_batch
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    # model = BertForSequenceClassification.from_pretrained(args.bert_model,
    #           cache_dir=cache_dir,
    #           num_labels=num_labels) # num_labels as proto_network's embedding size
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    print("Whether Initialize the model?")
    if args.is_init:
        print("Initializing........")
        model.apply(model.init_bert_weights)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                         lr=args.learning_rate,
    #                         warmup=args.warmup_proportion,
    #                         t_total=num_train_optimization_steps)
    optimizer = optim.Adam(model.parameters(), lr=args.inner_learning_rate)
    K_last = 3
    epsilon = 1e-6
    task_rewards = {}
    last_observation = {}
    # if N_class > 2:
    #     print('using CrossEntropyLoss() ...')
    #     # loss_fct = CrossEntropyLoss()
    # else:
    #     print('using BCELoss() ...')
        # loss_fct = BCELoss()
    # for task_id in range(train_task_number):
    #     task_rewards[task_id] = []
    #     last_observation[task_id] = 0

    if args.do_train:
        # task_list = np.random.permutation(np.arange(train_task_number)) # training data consists of several domains, each has t2, t4, t5 ==> nb of domains * 3

        train_examples = processor.train_examples

        for epoch in trange(int(meta_iters), desc="Iterations"): # meta-training
            # reward_prob = []
            # for task_id in task_list:
            #     if len(task_rewards[task_id]) == 0:
            #         reward_prob.append(1)
            #     elif len(task_rewards[task_id]) < K_last:
            #         reward_prob.append(abs(np.random.choice(task_rewards[task_id], 1)[0]))
            #     else:
            #         reward_prob.append(abs(np.random.choice(task_rewards[task_id][-K_last:], 1)[0]))
            # reward_prob = [math.exp(prob) for prob in reward_prob]
            # reward_prob = [float(prob)/sum(reward_prob) for prob in reward_prob]
            # reward_prob = get_train_prob(reward_prob, N_task,epsilon)
            # selected_tasks = np.random.choice(task_list, N_task,replace=False) # pick randomly N_task number of tasks
            weight_before = deepcopy(model.state_dict())
            update_vars = []
            fomaml_vars = []

            # the difference between this code and the code by openAI is that here we preselect N_task number of tasks
            # inside the outer loop, then in the second inner loop, we loop over each of these N_task tasks,
            # and, in each iteration in the third inner loop (num updates), we sample support and query sets from each
            # of the N_task(s) in the second loop

            # whereby in supervised_reptile code by Open AI, we DONT sample a preselected N_task, BUT, we define the
            # variable meta_batch_size, then in the second inner loop, we loop meta_batch_size number of times,
            # and in each iteration, we sample a 'mini_dataset', then in the third inner loop, we sample
            # a support and query set from each 'mini_dataset' in the second loop

            # in fact, N_shot and N_query are the inner batch size
            # don't confuse meta_batch_size, it 1 by default and no need to do a mini-dataset
            # and then sample mini batches from there, just sample mini-bathes directly

            # for task_id in tqdm(selected_tasks,desc="Task"): # for each task in randomly selected tasks -- this is the first loop over meta_batch_size
            #                                                  # https://github.com/openai/supervised-reptile/blob/8f2b71c67a31c1a605ced0cecb76db876b607a7a/supervised_reptile/reptile.py#L64

            for _ in range(meta_batch): # N_task is meta_batch_size

                # here, we sample a mini-dataset, from which we sample examples per class in the inner loop
                # https://github.com/openai/supervised-reptile/blob/8f2b71c67a31c1a605ced0cecb76db876b607a7a/supervised_reptile/reptile.py#L65

                mini_dataset = processor._sample_mini_dataset(examples=train_examples, num_classes=N_class, num_shots=num_shots)

                for batch in _mini_batches(samples=mini_dataset, batch_size=inner_batch, num_batches=inner_iters, replacement=False):  # for inner number of updates -- this is the inner loop over mini batches
                    # https://github.com/openai/supervised-reptile/blob/8f2b71c67a31c1a605ced0cecb76db876b607a7a/supervised_reptile/reptile.py#L66

                    inputs = batch
                    train_support = inputs
                    # train_support, train_query = processor.get_examples_per_task(B=inner_batch, N=N_class, K=num_shots, set_type='train')  # sample support and query sets from the selected task

                    support_features = convert_examples_to_features(
                        train_support, label_list, args.max_seq_length, tokenizer, output_mode)

                    support_input_ids = torch.tensor([f.input_ids for f in support_features], dtype=torch.long).to(device)
                    support_mask_ids = torch.tensor([f.input_mask for f in support_features], dtype=torch.long).to(device)
                    support_seg_ids = torch.tensor([f.segment_ids for f in support_features], dtype=torch.long).to(device)
                    if output_mode == "classification":
                        support_labels = torch.tensor([f.label_id for f in support_features], dtype=torch.long).to(device)
                    elif output_mode == "regression":
                        all_label_ids = torch.tensor([f.label_id for f in support_features], dtype=torch.float).to(device)
                    last_backup = deepcopy(model.state_dict())
                    model.train()
                    # define a new function to compute loss values for both output_modes
                    # logits = model(support_input_ids, support_seg_ids, support_mask_ids, labels=None)
                    outputs = model(support_input_ids, support_seg_ids, support_mask_ids, labels=None)
                    logits = outputs.logits
                    # print(logits.shape)
                    # print(support_labels.shape)
                    if output_mode == "classification":
                        # if N_class > 2:

                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, num_labels), support_labels.view(-1))

                        # else:
                        #     loss_fct = BCELoss()
                        #     loss = loss_fct(logits.view(-1, num_labels), torch.nn.functional.one_hot(support_labels).float())
                    elif output_mode == "regression":
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), support_labels.view(-1))

                    preds = logits.detach().cpu().numpy()

                    print('hello')

                    result = compute_metrics(task_name, np.argmax(preds, axis=1), support_labels.detach().cpu().numpy())
                    task_acc = result['acc']
                    task_f1 = result['f1']
                    task_prec = result['precision']
                    task_rec = result['recall']
                    print('epoch: {}, accuracy: {}, precision: {}, recall: {}, f1: {}, loss: {}'.format(epoch, task_acc,
                                                                                                        task_prec,
                                                                                                        task_rec,
                                                                                                        task_f1, loss))


                    if epoch % 10 == 0:
                        result = compute_metrics(task_name, np.argmax(preds, axis=1), support_labels.detach().cpu().numpy())
                        task_acc = result['acc']
                        task_f1 = result['f1']
                        task_prec = result['precision']
                        task_rec = result['recall']
                        print('epoch: {}, accuracy: {}, precision: {}, recall: {}, f1: {}, loss: {}'.format(epoch, task_acc, task_prec, task_rec, task_f1, loss))

                    # logger.info("Batch %d ,Accuracy: %s", step,result["acc"])

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                weight_after = deepcopy(model.state_dict())
                update_vars.append(weight_after)
                tmp_fomaml_var = {}

                # task_rewards[task_id].append(task_acc - last_observation[task_id])
                # last_observation[task_id] = task_acc

                if not Is_reptile:
                    for name in weight_after:
                        tmp_fomaml_var[name] = weight_after[name] - last_backup[name]
                    fomaml_vars.append(tmp_fomaml_var)
                model.load_state_dict(weight_before) # here we are: https://github.com/openai/supervised-reptile/blob/8f2b71c67a31c1a605ced0cecb76db876b607a7a/supervised_reptile/reptile.py#L72


            new_weight_dict = {}
            # print(weight_before)
            if Is_reptile:
                for name in weight_before:
                    weight_list = [tmp_weight_dict[name] for tmp_weight_dict in update_vars]
                    weight_shape = list(weight_list[0].size())
                    stack_shape = [len(weight_list)] + weight_shape
                    stack_weight = torch.empty(stack_shape)
                    for i in range(len(weight_list)):
                        stack_weight[i,:] = weight_list[i]
                    if device == "gpu":
                        new_weight_dict[name] = torch.mean(stack_weight, dim=0).cuda()
                    else:
                        new_weight_dict[name] = torch.mean(stack_weight, dim=0)
                    # new_weight_dict[name] = torch.mean(stack_weight, dim=0)
                    new_weight_dict[name] = weight_before[name]+(new_weight_dict[name]-weight_before[name])/args.inner_learning_rate*args.outer_learning_rate
            else:
                for name in weight_before: 
                    weight_list = [tmp_weight_dict[name] for tmp_weight_dict in fomaml_vars]
                    weight_shape = list(weight_list[0].size())
                    stack_shape = [len(weight_list)] + weight_shape
                    stack_weight = torch.empty(stack_shape)
                    for i in range(len(weight_list)):
                        stack_weight[i,:] = weight_list[i]
                    if device == "gpu":
                        new_weight_dict[name] = torch.mean(stack_weight, dim=0).cuda()
                    else:
                        new_weight_dict[name] = torch.mean(stack_weight, dim=0)
                    # new_weight_dict[name] = torch.mean(stack_weight, dim=0).cuda()
                    # new_weight_dict[name] = torch.mean(stack_weight, dim=0)
                    new_weight_dict[name] = weight_before[name].to(device)+new_weight_dict[name].to(device)/args.inner_learning_rate*args.outer_learning_rate
            model.load_state_dict(new_weight_dict)
            # ###MAML code
            # update_lr = 5e-2
            # for i in range(task_num):
            #     logits = model(input_ids, segment_ids, input_mask, labels=None)
            #     if output_mode = "classification":
            #         loss_fct = CrossEntropyLoss()
            #         loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            #         grad = torch.autograd.grad(loss, model.parameters())
            #         fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, model.parameters())))
            #         with torch.no_grad():
                        # logits

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) :
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)


    loss_list = {}
    global_step = 0
    nb_tr_steps = 0
    Meta_optimizer = optim.Adam(model.parameters(), lr=args.FSL_learning_rate)

    # in supervised reptile, evaluation time, we run first outer loop with eval_iterations times
    # then, in each iteration, we sample a minidataset. We split the minidataset into
    # train and test sets ==> the test set is by default 1 example per class taken from train.
    # then in the second inner loop, we sample mini batches from the train set (inner iters)
    # number of times and we forward to the network, calculate gradients and update weights.
    # after that comes the prediction (evaluation) phase:
    # if transductive: we forward the test set (from outer loop) and calculate results
    # if not transductive: we add the training set to each sample in the test set
    # , forward to network, and calculate results for each group

    # in this code, NO eval_iterations, we have a list of tasks, this is the outer loop
    # then in inner loop, we have a set of support examples for each task id
    # we forward the support sets to the network, calc gradients, and update weights
    # then after inner loop is done, we have a set of eval_examples. We convert them
    # to features, put them in Sequential Data Loader, evaluate results
    # then reload old weights back again
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        weight_before = deepcopy(model.state_dict())
        # for task_id in trange(fsl_task_number, desc="Task"):  # predefine number of tasks as 'support'
        dev_examples = processor.dev_examples
        model.train()
        mini_dataset = processor._sample_mini_dataset(examples=dev_examples, num_classes=N_class, num_shots=num_shots)
        # for _ in range(args.FSL_epochs):  # start the outer loop of meta eval iterations (but no its 1 by default)
        for batch in _mini_batches(samples=mini_dataset, batch_size=inner_batch, num_batches=inner_iters, replacement=False):

            # support_examples = processor.get_fsl_support(args.data_dir, task_id)
            # support_examples, _ = processor.get_examples_per_task(train_batch_s, N_class, N_shot, N_query, 'dev')
            support_examples = batch
            support_features = convert_examples_to_features(
                support_examples, label_list, args.max_seq_length, tokenizer, output_mode
            )
            support_input = torch.tensor([f.input_ids for f in support_features], dtype=torch.long).to(device)
            support_mask = torch.tensor([f.input_mask for f in support_features], dtype=torch.long).to(device)
            support_seg = torch.tensor([f.segment_ids for f in support_features], dtype=torch.long).to(device)
            support_labels = torch.tensor([f.label_id for f in support_features], dtype=torch.long).to(device)

            outputs = model(support_input, support_seg, support_mask, labels=None)
            logits = outputs.logits
            # if N_class > 2:

            loss = CrossEntropyLoss()
            loss = loss(logits.view(-1, num_labels), support_labels.view(-1))

            # else:
            #     loss = BCELoss()
            #     loss = loss(logits.view(-1, num_labels), torch.nn.functional.one_hot(support_labels).float())
            loss.backward()  # calculate gradients and update weights
            # print("Current loss: ", loss)
            # print("fsl training!")
            Meta_optimizer.step()
            Meta_optimizer.zero_grad()

        # eval_examples, _ = processor.get_fsl_test_examples(args.data_dir, task_id)  # get test examples that are query
        eval_examples, _ = processor.get_test_tasks()
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                # logits = model(input_ids, segment_ids, input_mask, labels=None)
                outputs = model(input_ids, segment_ids, input_mask, labels=None)
                logits = outputs.logits

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                # if N_class > 2:

                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                # else:
                #     loss_fct = BCELoss()
                #     tmp_eval_loss = loss_fct(logits.view(-1, num_labels), torch.nn.functional.one_hot(label_ids).float())

                # loss_fct = CrossEntropyLoss()
                # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        softmax_output = preds

        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        # print("Index    Prediction    Labels    softmax_output")
        # if (task_id - 1) % 3 == 0:
        #     for i in range(len(preds)):
        #         if preds[i] != all_label_ids.numpy()[i]:
        #             print("Wrong Prediction! ")
        #             print(i, "    ", preds[i], "    ", all_label_ids.numpy()[i], softmax_output[i])
        #             print(eval_examples[i].text_a)
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        # loss = tr_loss/nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        # loss_list[task_id] = result['acc']
        # result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        model.load_state_dict(weight_before)

        print('\nClassification report:\n{}\n'.format(classification_report(all_label_ids.numpy(), preds, target_names=label_list)))
        print('\nConfusion Matrix\n{}\n'.format(confusion_matrix(all_label_ids.numpy(), preds)))
        # for id, acc in loss_list.items():
        #     print("Task id: ", id, " ---- acc: ", acc)
        # print("Average acc is: ", np.mean(list(loss_list.values())))


if __name__ == "__main__":
    main()

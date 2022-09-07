import glob
import os
import csv
import logging
import pandas as pd
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from arabert.preprocess import ArabertPreprocessor
from sklearn import metrics
import time
import datetime

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

# from transformers import BertTokenizer
# from transformers import BertForSequenceClassification
# from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME


logging.basicConfig(format='%(message)s', #"format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample:

    def __init__(self, text_a, text_b=None, label=None, guid=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.guid = guid


class InputFeatures:

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor:

    def __init__(self, model_name, text_col, label_col):
        self.labels = None
        self.model_name = model_name
        self.text_col = text_col
        self.label_col = label_col

    def _read_dataset(self, df_path):
        df = pd.read_csv(df_path) if '.csv' in df_path else pd.read_excel(df_path)
        arabert_prep = ArabertPreprocessor(model_name=self.model_name)
        if self.labels is None:
            sentences = []
            labels = set()
            for i, row in df.iterrows():
                sentence = str(row[self.text_col])
                if sentence.strip().isdigit():
                    continue
                # according to https://github.com/aub-mind/arabert#preprocessing
                # It is recommended to apply our preprocessing function before training/testing on any dataset
                sentence = arabert_prep.preprocess(sentence)
                label = row[self.label_col]

                labels.add(label)
                sentences.append([sentence, label])

            self.labels = list(labels)
        else:
            sentences = []
            for i, row in df.iterrows():
                sentence = str(row[self.text_col])
                label = row[self.label_col]

                if sentence.strip().isdigit():
                    continue

                sentences.append([sentence, label])

        return sentences

    def get_train_examples(self, df_path):
        print('reading the training dataset from {} ...'.format(df_path))
        return self._create_examples(self._read_dataset(df_path=df_path), "train")

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
        return InputExample(text_a=sentence, label=label, guid=guid)

    def get_labels(self):
        """ gets the list of unique labels. Called only after calling _read_dataset()"""
        return self.labels


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    tokens_a_longer_max_seq_length = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        len_tokens_a = len(tokens_a)
        len_tokens_b = 0

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            len_tokens_b = len(tokens_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        if (len_tokens_a + len_tokens_b) > (max_seq_length - 2):
            tokens_a_longer_max_seq_length += 1

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
        # since the [SEP] token unambigiously separates the sequences, but it makes
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

        assert len(input_ids)==max_seq_length
        assert len(input_mask)==max_seq_length
        assert len(segment_ids)==max_seq_length

        label_id = label_map[example.label]
        if ex_index < 1 and example.guid is not None and example.guid.startswith('train'):
            logger.info("\n\n*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("\n\n")

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    logger.info(":: Sentences longer than max_sequence_length: %d" % (tokens_a_longer_max_seq_length))
    logger.info(":: Num sentences: %d" % (len(examples)))
    return features, label_map


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


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--data_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--bert_model", default="aubmindlab/bert-base-arabertv2", type=str, required=True,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                          "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
    #                          "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="aubmindlab/bert-base-arabertv2", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")

    # parser.add_argument("--task_name",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The name of the task to train.")
    #
    parser.add_argument("--task_name",
                        default="classification_arabert",
                        type=str,
                        help="The name of the task to train.")

    # parser.add_argument("--output_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir",
                        default="bert_output/",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # parser.add_argument("--do_train",
    #                     action='store_true',
    #                     help="Whether to run training.")
    parser.add_argument("--do_train",
                        action='store_false',
                        help="Whether to run training.")
    # parser.add_argument("--do_eval",
    #                     action='store_true',
    #                     help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval",
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
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        # default=2.0,
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

    parser.add_argument("--train_set",
                        # default="input/Discourse_Profiling/df_train_cleaned.xlsx",
                        default="input/corpus-webis-editorials-16/df_train.xlsx",
                        type=str,
                        help="path to the training dataset.")

    parser.add_argument("--dev_set",
                        # default="input/Discourse_Profiling/df_dev_cleaned.xlsx",
                        default="input/corpus-webis-editorials-16/df_dev.xlsx",
                        type=str,
                        help="path to the training dataset.")

    parser.add_argument("--test_set",
                        # default="input/Discourse_Profiling/df_test_cleaned.xlsx",
                        default="input/corpus-webis-editorials-16/df_test.xlsx",
                        type=str,
                        help="path to the testing dataset.")

    parser.add_argument("--text_column", default="Sentence_ar", type=str, help="Name of the column that contains text data")
    parser.add_argument("--label_column", default="Label", type=str, help="Name of the column that contains the labels")

    args = parser.parse_args()

    if args.test_set is not None:
        logger.info("Test set: "+args.test_set)

    if args.local_rank==-1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank!=-1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'parameters.txt'), 'w') as fOut:
        fOut.write(str(args))

    task_name = args.task_name.lower()

    # if task_name not in processors:
    #     raise ValueError("Task not found: %s" % (task_name))
    #
    # processor = processors[task_name](binarize_labels, use_all_data)
    processor = DataProcessor(model_name=args.bert_model, text_col=args.text_column, label_col=args.label_column)
    # label_list = processor.get_labels()
    # num_labels = len(label_list)

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        ttemp = time.time()
        train_examples = processor.get_train_examples(args.train_set)
        # train_examples = train_examples[:100]
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        print('loaded training examples - took {}'.format(format_time(time.time() - ttemp)))
    # Prepare model
    # model = BertForSequenceClassification.from_pretrained(args.bert_model,
    #                                                       cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
    #                                                           args.local_rank),
    #                                                       num_labels=num_labels)

    label_list = processor.get_labels() # we can call get_labels() because we called get_train_examples()
    num_labels = len(label_list)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    print('loaded BERT model for sequence classification ...')
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank!=-1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank!=-1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale==0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        # optimizer = BertAdam(optimizer_grouped_parameters,
        #                      lr=args.learning_rate,
        #                      warmup=args.warmup_proportion,
        #                      t_total=t_total)

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    t1 = time.time()
    if args.do_train:
        with open(os.path.join(args.output_dir, "train_sentences.csv"), "w", encoding="utf-8") as writer:
            for idx, example in enumerate(train_examples):
                writer.write("%s\t%s\t%s\n" % (example.label, example.text_a, example.text_b))


        train_features, label_map = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Labels = %s", ", ".join(label_list))
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank==-1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            print('epoch: {}'.format(epoch))
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
                loss = outputs.loss
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps==0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if args.do_eval:
                eval_results_filename = "test_results_epoch_%d.txt" % (epoch)
                eval_prediction_filename = "test_predictions_epoch_%d.txt" % (epoch)
                do_evaluation(processor, args, label_list, tokenizer, model, device, tr_loss, nb_tr_steps, global_step,
                              task_name, eval_results_filename, eval_prediction_filename, mode='validation', label_map=label_map)

        elapsed = format_time(time.time() - t1)
        print('time elapsed to complete training + validation: {}'.format(elapsed))

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)


    if args.do_eval and (args.local_rank==-1 or torch.distributed.get_rank()==0):
        eval_results_filename = "test_results.txt"
        eval_prediction_filename = "test_predictions.txt"
        do_evaluation(processor, args, label_list, tokenizer, model, device, tr_loss, nb_tr_steps, global_step, task_name, eval_results_filename,
                      eval_prediction_filename, mode='testing', label_map=label_map)

    elapsed = format_time(time.time() - t1)
    print('time elapsed to complete training + testing: {}'.format(elapsed))



def do_evaluation(processor, args, label_list, tokenizer, model, device, tr_loss, nb_tr_steps, global_step, task_name, eval_results_filename,
                  eval_prediction_filename, mode, label_map):
    if mode == 'validation':
        eval_examples = processor.get_test_examples(args.dev_set)
    else:
        eval_examples = processor.get_test_examples(args.test_set)
    eval_examples = eval_examples
    eval_features, _ = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    predicted_labels = []
    predicted_labels_ints = []
    gold_labels_ints = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        for prediction in np.argmax(logits, axis=1):
            predicted_labels.append(label_list[prediction])
            predicted_labels_ints.append(prediction)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1


    eval_accuracy = eval_accuracy / nb_eval_examples
    loss = tr_loss / nb_tr_steps if args.do_train else None
    result = {
              'eval_accuracy': eval_accuracy,
              'global_step': global_step,
              'train_loss': loss}

    output_eval_file = os.path.join(args.output_dir, eval_results_filename)
    with open(output_eval_file, "w") as writer:
        logger.info("\n\n\n***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info("\n\n\n")

    # print(classification_report([e.label for e in eval_examples], predicted_labels, target_names=labels))

    output_pred_file = os.path.join(args.output_dir, eval_prediction_filename)
    with open(output_pred_file, "w", encoding="utf-8") as writer:
        y_test, y_pred = [], []
        for idx, example in enumerate(eval_examples):
            gold_label = example.label
            pred_label = predicted_labels[idx]

            y_test.append(label_map[gold_label])
            y_pred.append(label_map[pred_label])

            text_a = example.text_a.replace("\n", " ")
            text_b = example.text_b.replace("\n", " ") if example.text_b is not None else "None"

            writer.write("%s\t%s\t%s\t%s\n" % (gold_label, pred_label, text_a, text_b))

        report = metrics.classification_report(y_test, y_pred, target_names=list(label_map.keys()), labels=list(range(len(label_map)+1)), digits=4)
        confusion = metrics.confusion_matrix(y_test, y_pred)

        print(report)
        print(confusion)

if __name__=="__main__":
    main()
from transformers import BertForSequenceClassification, BertConfig
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


logging.basicConfig(format='%(message)s', #"format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class InputExample:

    def __init__(self, text_a, text_b=None, guid=None):
        self.text_a = text_a
        self.text_b = text_b
        self.guid = guid


class InputFeatures:

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class DataProcessor:

    def __init__(self, model_name, text_col):
        self.labels = None
        self.model_name = model_name
        self.text_col = text_col

    def _read_dataset(self, df_path):
        df = pd.read_csv(df_path) if '.csv' in df_path else pd.read_excel(df_path)
        arabert_prep = ArabertPreprocessor(model_name=self.model_name)
        sentences = []
        count = 0
        for i, row in df.iterrows():
            if count == 1000:
                break
            sentence = str(row[self.text_col])
            if sentence.strip().isdigit():
                continue
            # according to https://github.com/aub-mind/arabert#preprocessing
            # It is recommended to apply our preprocessing function before training/testing on any dataset
            sentence = arabert_prep.preprocess(sentence)
            sentences.append(sentence)
            count += 1

        return sentences

    def get_test_examples(self, df_path):
        print('reading the testing dataset from {} ...'.format(df_path))
        return self._create_examples(self._read_dataset(df_path=df_path), "test")

    def _create_examples(self, data_tuples, set_type):
        """ Creates examples for the train and test sets """
        exampels = []

        for i, sentence in enumerate(data_tuples):
            guid = "%s-%s" % (set_type, i)
            exampels.append(self._get_input_example(guid=guid, sentence=sentence))

        return exampels

    def _get_input_example(self, guid, sentence):
        return InputExample(text_a=sentence, guid=guid)


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

        # label_id = label_map[example.label]
        if ex_index < 1 and example.guid is not None and example.guid.startswith('train'):
            logger.info("\n\n*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("\n\n")

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))

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


def get_label_mapping(train_set, label_col):
    labels = set()
    train_df = pd.read_csv(train_set) if '.csv' in train_set else pd.read_excel(train_set)
    for i, row in train_df.iterrows():
        labels.add(row[label_col])

    label_list = list(labels)
    label_map = {label: i for i, label in enumerate(label_list)}
    return label_map


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--bert_model", default="aubmindlab/bert-base-arabertv2", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--state_dict_path", default="bert_output/corpus-webis-editorials-16/pytorch_model.bin")
    #
    parser.add_argument("--task_name",
                        default="classification_arabert",
                        type=str,
                        help="The name of the task to train.")

    parser.add_argument("--output_dir",
                        default="bert_predictions/nahar/",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

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

    parser.add_argument("--test_set",
                        # default="input/Discourse_Profiling/df_test_cleaned.xlsx",
                        # default="input/corpus-webis-editorials-16/df_test.xlsx",
                        default="testing_datasets_discourse/nahar/df_test_2007.xlsx",
                        type=str,
                        help="path to the testing dataset.")

    parser.add_argument("--text_column", default="Sentence", type=str,
                        help="Name of the column that contains text data in the **testing dataset**")
    parser.add_argument("--label_column", default="Label", type=str, help="Name of the column that contains the labels in the **training dataset**")

    args = parser.parse_args()

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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'parameters.txt'), 'w') as fOut:
        fOut.write(str(args))

    task_name = args.task_name.lower()

    processor = DataProcessor(model_name=args.bert_model, text_col=args.text_column)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # label_list = processor.get_labels()  # we can call get_labels() because we called get_train_examples()
    # num_labels = len(label_list)

    # model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    config = BertConfig.from_pretrained(args.bert_model, num_labels=8)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, config=config)
    model.load_state_dict(torch.load(args.state_dict_path))
    print('loaded BERT model for sequence classification ...')

    if args.fp16:
        model.half()

    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    eval_results_filename = "test_results.txt"
    eval_prediction_filename = "test_predictions.txt"
    label_map = get_label_mapping(train_set=args.train_set, label_col=args.label_column)
    label_list = list(label_map.keys())
    do_evaluation(processor, args, label_list, tokenizer, model, device, task_name,
                  eval_results_filename, eval_prediction_filename, mode='testing', label_map=label_map)


def do_evaluation(processor, args, label_list, tokenizer, model, device, task_name, eval_results_filename,
                  eval_prediction_filename, mode, label_map):

    eval_examples = processor.get_test_examples(args.test_set)

    eval_features, _ = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0

    predicted_labels = []

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()

        for prediction in np.argmax(logits, axis=1):
            predicted_labels.append(label_list[prediction])

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    # test_df = pd.read_csv(args.test_set) if '.csv' in args.test_set else pd.read_excel(args.test_set)
    test_df = pd.DataFrame(columns=['Sentence', 'Prediction'])
    sentences, predictions = [], []
    output_pred_file = os.path.join(args.output_dir, eval_prediction_filename)
    with open(output_pred_file, "w", encoding="utf-8") as writer:
        y_pred = []
        for idx, example in enumerate(eval_examples):
            # gold_label = example.label
            pred_label = predicted_labels[idx]

            # y_test.append(label_map[gold_label])
            y_pred.append(label_map[pred_label])

            text_a = example.text_a.replace("\n", " ")
            text_b = example.text_b.replace("\n", " ") if example.text_b is not None else "None"

            # writer.write("%s\t%s\t%s\t%s\n" % (gold_label, pred_label, text_a, text_b))
            writer.write("%s\t%s\t%s\n" % (pred_label, tokenizer.unpreprocess(text_a), text_b))

            sentences.append(tokenizer.unpreprocess(text_a))
            predictions.append(pred_label)
        test_df['Sentence'] = sentences
        test_df['Prediction'] = predictions

        test_df.to_excel(os.path.join(args.output_dir, args.test_set.split('/')[-1]))


if __name__ == '__main__':
    main()



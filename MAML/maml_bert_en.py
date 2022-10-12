# coding=utf-8
from __future__ import absolute_import, division, print_function

"""'update version of run_glue.py from huggingface"""
""" Meta_Learning and Finetuning the  models for sequence classification on XNLI (Bert, XLM, XLNet, RoBERTa)."""
import argparse
import logging
import os
import random
import pickle
import numpy as np
from collections import defaultdict
import torch
import torch.optim as optim

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler,
                              TensorDataset, ConcatDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import higher
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
# need xnli_convert_example_to_features here

logger = logging.getLogger(__name__)
import itertools
from utils_nlp import processors as processors
from utils_nlp import output_modes as output_modes
from utils_nlp import convert_examples_to_features
from sklearn.metrics import *
import pandas as pd

def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def report_memory(name=''):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_cached()/ mega_bytes)
    print(string)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def select_auxilary_tasks(args,lang_maps,n):
    random.seed(np.random.randint(1,1000))
    langs=np.random.choice(list(lang_maps.keys()), n, replace=False)
    set_seed(args)
    return langs


def sample_indecies(args,dataset,n):
    np.random.seed(np.random.randint(1, 1000))
    sample_indecies=np.random.choice(range(len(dataset)), args.train_batch_size*n, replace=False)
    set_seed(args)
    return sample_indecies


def sample_tasks(args, features):
    tasks = []
    # sample n examples per class
    for cls in range(args.num_classes):
        # get all features having this class
        features_cls = [f for f in features if f.label == cls]
        # sample a random set of K examples per this class
        sample = random.sample(features_cls, args.n)  # K  for K shot classification
        tasks.extend(sample)
    random.shuffle(tasks)
    return tasks


def make_data_tensor(args, features):
    num_total_batches = 1000
    all_data = []
    for ifold in range(num_total_batches):
        task = sample_tasks(args=args, features=features)
        all_data.extend(task)

    all_data_batches = []
    examples_per_batch = args.num_classes * args.n
    for i in range(args.per_gpu_train_batch_size):
        data_batch = all_data[i * examples_per_batch:(i + 1) * examples_per_batch]
        all_data_batches.append(data_batch)
    return all_data_batches


def maml(args, model, dataset, tokenizer, suffix=None):
    print('started training ...')
    model.train()

    # args.output_dir_meta = os.path.join(args.output_dir_meta, args.model_name_or_path, suffix)

    if not os.path.exists(args.output_dir_meta) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir_meta)

    # Loop to handle tasks in XNLI
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for mt_itr in range(1, args.meta_learn_iter + 1):  # outer loop learning

        batch_of_tasks = make_data_tensor(args=args, features=dataset)

        losses = []

        for task in tqdm(batch_of_tasks):
            support_features = task[:int(args.per_gpu_train_batch_size/2)]
            query_features = task[int(args.per_gpu_train_batch_size/2):]

            support_input_ids = torch.tensor([f.input_ids for f in support_features], dtype=torch.long).to(args.device)
            support_mask_ids = torch.tensor([f.input_mask for f in support_features], dtype=torch.long).to(args.device)
            support_seg_ids = torch.tensor([f.segment_ids for f in support_features], dtype=torch.long).to(args.device)
            support_labels = torch.tensor([f.label for f in support_features], dtype=torch.long).to(args.device)

            query_input_ids = torch.tensor([f.input_ids for f in query_features], dtype=torch.long).to(args.device)
            query_mask_ids = torch.tensor([f.input_mask for f in query_features], dtype=torch.long).to(args.device)
            query_seg_ids = torch.tensor([f.segment_ids for f in query_features], dtype=torch.long).to(args.device)
            query_labels = torch.tensor([f.label for f in query_features], dtype=torch.long).to(args.device)

            inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)

            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fast_model, diffopt):
                for itr in range(args.inner_train_steps):
                    fast_model.train()
                    # print("-------------")
                    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

                    # batch_valid=tuple(t.to(args.device) for t in next(iter(task_valid_dataloader)))
                    # n_meta_lr = int((args.train_batch_size) / 2)

                    input_fast_train = {'input_ids': support_input_ids,
                                        'attention_mask': support_mask_ids,
                                        'labels': support_labels}

                    if args.model_type != 'distilbert':
                        input_fast_train['token_type_ids'] = support_seg_ids if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                    outputs = fast_model(**input_fast_train)
                    loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            diffopt.step(scaled_loss)
                    else:
                        diffopt.step(loss)

                input_fast_valid = {'input_ids': query_input_ids,
                                    'attention_mask': query_mask_ids,
                                    'labels': query_labels}
                if args.model_type != 'distilbert':
                    input_fast_valid['token_type_ids'] = query_seg_ids if args.model_type in ['bert',
                                                                                              'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = fast_model(**input_fast_valid)
                qry_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                qry_loss.backward()
                losses.append(qry_loss.item())

        print('average loss: {}'.format(np.mean(losses)))

        optimizer.step()
        model.zero_grad()
        logger.info("meta-learning it %s", mt_itr)

    logger.info("Saving model checkpoint to %s", args.output_dir_meta)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model,
                                            'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir_meta)
    tokenizer.save_pretrained(args.output_dir_meta)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir_meta, 'training_args.bin'))

    return model

def get_results(labels, preds):
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


def evaluate_task(args, model, eval_dataset, label_list, early=False, prefix=""):
    eval_task =  args.task_name
    # eval_output_dir = os.path.join(args.eval_task_dir,args.model_name_or_path,prefix)
    eval_output_dir = args.eval_task_dir
    results = {}

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # @todo: take english:
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    # result = compute_metrics(eval_task, preds, out_label_ids)
    # results.update(result)

    result = get_results(labels=out_label_ids, preds=preds)
    result['eval_loss'] = eval_loss
    report = classification_report(out_label_ids, preds, target_names=label_list)
    confusion = confusion_matrix(out_label_ids, preds)

    for k in result:
        print('{}: {}'.format(k, result[k]))
    print(report)
    print(confusion)

    print('saving results to {} ....'.format(os.path.join(eval_output_dir, 'test_actual_predicted.csv')))
    label_map = {i: label for i, label in enumerate(label_list)}
    df = pd.DataFrame()
    df['actual'] = [label_map[idx] for idx in out_label_ids]
    df['predicted'] = [label_map[idx] for idx in preds]
    df.to_csv(os.path.join(eval_output_dir, 'test_actual_predicted.csv'), index=False)
    print('Done saving results')

    if not early:
        #output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        #with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
                #writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="../../data/XNLI-1.0", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    parser.add_argument("--train_data", default="input/df_en.csv", type=str, help="path to the .csv training dataset")
    parser.add_argument("--test_data",  default="input/sentences_ocr_corrected_discourse_profiling_en.csv", type=str, help="path to the .csv testing  dataset")

    parser.add_argument("--model_type", default='BERT', type=str,
                        help="Model type selected in the list: ")
    # parser.add_argument("--model_name_or_path", default='', type=str,
    #                     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
    #                         ALL_MODELS))
    parser.add_argument("--model_name_or_path", default='', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list:")

    parser.add_argument("--task_name", default='dp', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))

    parser.add_argument("--cache_dir", default='../../data/BERT/meta_learning' , type=str,
                        help="The  directory where the pre-train model is .")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")


    parser.add_argument("--num_classes", default=8, type=int, help="number of classes")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # meta-learning setup:

    parser.add_argument("--fine_tune", action='store_true',
                        help="Whether to run fine-tuning language based.")

    parser.add_argument("--do_maml", action='store_true',
                        help="Whether to run maml.")

    parser.add_argument("--do_maml_on_train", action='store_true',
                        help="Whether to run maml on MNLI train dataset.")

    parser.add_argument("--do_reptile", action='store_true',
                        help="Whether to run reptile.")

    parser.add_argument("--do_augment", action='store_true',
                        help="Whether to run augment (it means not maml).")

    parser.add_argument("--bi_auxs", action='store_true',
                        help="Whether to run augment (it means not maml).")
    parser.add_argument("--do_fine_tune_tasks",
                        action='store_true')
    parser.add_argument("--early_stopping",
                        action='store_true')

    parser.add_argument('--meta_learn_iter', default=2, type=int,
                        help='Number of iteration to perform on meta-learning')

    parser.add_argument('--n', default=2, type=int, help='Support samples per class for few-shot tasks')

    parser.add_argument('--inner_train_steps', default=3, type=int,
                        help='Number of inner-loop updates to perform on training tasks')
    parser.add_argument('--inner_lr', default=1e-4, type=float)

    parser.add_argument("--output_dir_meta", default='meta_learning/meta/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir_fine_tune", default='meta_learning/task_fine_tune/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_task_dir", default='meta_learning/task_fine_tune/eval', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--target_lang", type=str,
                        default="es")
    parser.add_argument("--percent", type=int,
                        default=100)

    parser.add_argument("--target", type=str)
    parser.add_argument("--auxiliary", type=int,
                        default=0)

    parser.add_argument("--combinations", type=str,
                        default="", help="fr,es-bg,tr")
    ## early args
    parser.add_argument("--patience", type=int , default=5)

    parser.add_argument("--lang_dev_early", type=int,
                        default=1)


    args = parser.parse_args()
    if args.combinations!='':
        args.combinations= [ls.strip().split(",") for ls in args.combinations.split("-") ]
    args.cache_dir=os.path.join(args.cache_dir, str(args.percent))
    fin_tune_task="fin-True" if args.do_fine_tune_tasks else "fin-False"
    args.output_dir_meta=os.path.join(args.output_dir_meta, str(args.percent), fin_tune_task)

    args.config_name=os.path.join(args.cache_dir,"config.json")


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    processor.load_all_data(df_train_path=args.train_data,
                            df_dev_path=None,
                            df_test_path=args.test_data)

    train_examples = processor.train_examples
    test_examples = processor.test_examples


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    # tokenizer = BertTokenizer.from_pretrained(args.cache_dir, do_lower_case=args.do_lower_case)
    # config = BertConfig.from_pretrained(args.config_name,
    #                                       num_labels=num_labels,
    #                                       finetuning_task=args.task_name,
    #                                       cache_dir=args.cache_dir if args.cache_dir else None)

    # model = BertForSequenceClassification(config)
    # model.from_pretrained(args.cache_dir)

    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    training_dataset = convert_examples_to_features(examples=train_examples, label_list=label_list,
                                                    max_seq_length=args.max_seq_length, tokenizer=tokenizer,
                                                    output_mode="classification")
    # examples, label_list, max_seq_length, tokenizer, output_mode
    model = maml(args=args, model=model, dataset=training_dataset, tokenizer=tokenizer, suffix=None)

    # Evaluation
    testing_features = convert_examples_to_features(examples=test_examples, label_list=label_list,
                                                   max_seq_length=args.max_seq_length, tokenizer=tokenizer,
                                                   output_mode="classification")

    all_input_ids = torch.tensor([f.input_ids for f in testing_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.input_mask for f in testing_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.segment_ids for f in testing_features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in testing_features], dtype=torch.long)

    testing_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    evaluate_task(args, model, testing_dataset, label_list=label_list, prefix="")



if __name__ == "__main__":
    main()
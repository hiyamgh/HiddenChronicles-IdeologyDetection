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
    AutoTokenizer,
    BertForSequenceClassification, AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
# need xnli_convert_example_to_features here

logger = logging.getLogger(__name__)
import itertools
from utils_xmaml import processors as processors
from utils_xmaml import output_modes as output_modes
from utils_xmaml import convert_examples_to_features, load_cache_examples
from utils_xmaml import datasets

from sklearn.metrics import *
import pandas as pd
from tqdm import tqdm


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


# def select_auxilary_tasks(args,lang_maps,n):
#     random.seed(np.random.randint(1,1000))
#     langs=np.random.choice(list(lang_maps.keys()), n, replace=False)
#     set_seed(args)
#     return langs


def x_maml_with_aux_langs(args, model, lang_datasets, tokenizer, validation_dataset=None, suffix=None):
    model.train()

    args.output_dir_meta = os.path.join(args.output_dir_meta)

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

    validation_losses, validation_accuracies = [], []
    for mt_itr in tqdm(range(1, args.meta_learn_iter + 1)):  # outer loop learning
        inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
        losses = []
        for lang_id, dataset in lang_datasets.items():
            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            #task_train_samples, task_val_samples = torch.utils.data.random_split(dataset,[2000, len(dataset)-2000])
            task_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
            task_dataloader = DataLoader(dataset,sampler=task_sampler, batch_size=int(args.train_batch_size))

            #epoch_iterator = tqdm(task_dataloader, desc="Iteration",
            #                      disable=args.local_rank not in [-1, 0])
            # create bag of tasks from auxilary lang.
            for step, batch in enumerate(task_dataloader):
                if step!=mt_itr-1:
                    continue
            #batch=next(iter(task_dataloader))
                with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fast_model, diffopt):

                    fast_model = fast_model.to(args.device)

                    for itr in range(args.inner_train_steps):
                        fast_model.train()
                        #print("-------------")
                        set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
                        # batch = tuple(t.to(args.device) for t in batch)
                        batch = tuple(t.to(args.local_rank) for t in batch)
                        # batch_valid=tuple(t.to(args.device) for t in next(iter(task_valid_dataloader)))
                        n_meta_lr = int((args.train_batch_size)/2)
                        input_fast_train = {'input_ids': batch[0][:n_meta_lr],
                                            'attention_mask': batch[1][:n_meta_lr],
                                            'labels': batch[3][:n_meta_lr]}

                        # input_fast_train = input_fast_train.to(args.device)
                        # input_fast_train = {'input_ids': batch[0],
                        #                     'attention_mask': batch[1],
                        #                     'labels': batch[3]}

                        if args.model_type != 'distilbert':
                            input_fast_train['token_type_ids'] = batch[2][:n_meta_lr] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                        for key, value in input_fast_train.items():
                            input_fast_train[key] = input_fast_train[key].to(args.device)

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


                    input_fast_valid={'input_ids': batch[0][n_meta_lr:],
                                            'attention_mask': batch[1][n_meta_lr:],
                                            'labels': batch[3][n_meta_lr:]}
                    if args.model_type != 'distilbert':
                        input_fast_valid['token_type_ids'] = batch[2][n_meta_lr:] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                    for key, value in input_fast_valid.items():
                        input_fast_valid[key] = input_fast_valid[key].to(args.device)

                    outputs = fast_model(**input_fast_valid)
                    qry_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                    qry_loss.backward()
                    losses.append(qry_loss.item())

        # print('average loss: {}'.format(np.mean(losses)))
        optimizer.step()
        model.zero_grad()
        logger.info("meta-learning it {} - avg loss: {}".format(mt_itr, np.mean(losses)))

        if validation_dataset is not None and args.local_rank == 0:
            res = validate_task(args=args, model=model, val_dataset=validation_dataset)
            validation_accuracies.append(res['acc'])
            validation_losses.append(res['eval_loss'])

    if validation_dataset is not None and args.local_rank == 0:
        with open(os.path.join(args.output_dir_meta, 'validation_losses.pkl'), 'wb') as f:
            pickle.dump(validation_losses, f)

        with open(os.path.join(args.output_dir_meta, 'validation_accuracies.pkl'), 'wb') as f:
            pickle.dump(validation_accuracies, f)

    if args.local_rank == 0:
        logger.info("Saving model checkpoint to %s", args.output_dir_meta)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir_meta)
        tokenizer.save_pretrained(args.output_dir_meta)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir_meta, 'training_args.bin'))

    return model

def fine_tune_task(args, model, train_dataset, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)


    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
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

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            inputs = inputs.to(args.device)

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if args.tpu:
                args.xla_model.optimizer_step(optimizer, barrier=True)
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step, model


def validate_task(args, model, val_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # @todo: take english:
    eval_sampler = SequentialSampler(val_dataset) if args.local_rank == -1 else DistributedSampler(val_dataset)
    eval_dataloader = DataLoader(val_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running validation {} *****")
    logger.info("  Num examples = %d", len(val_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Validating"):
        model.eval()
        # batch = tuple(t.to(args.device) for t in batch)
        batch = tuple(t.to(args.local_rank) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            for key, value in inputs.items():
                inputs[key] = inputs[key].to(args.device)

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

    result = get_accuracy(labels=out_label_ids, preds=preds)
    result['eval_loss'] = eval_loss

    logger.info("Validation loss: {} - Validation accuracy: {}".format(eval_loss, result["acc"]))

    return result


def evaluate_task(args, model, eval_dataset, label_list, early=False, prefix=""):
    eval_task =  args.task_name
    eval_output_dir = os.path.join(args.eval_task_dir,args.model_name_or_path,prefix)
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

    if len(label_list) == 2:
        result = get_results(labels=out_label_ids, preds=preds, binary=True)
    else:
        result = get_results(labels=out_label_ids, preds=preds)

    result['eval_loss'] = eval_loss
    report = classification_report(out_label_ids, preds, labels=list(range(len(label_list))), target_names=label_list)
    confusion = confusion_matrix(out_label_ids, preds)

    for k in result:
        print('{}: {}'.format(k, result[k]))
    print(report)
    print(confusion)

    print('saving results to {} ....'.format(os.path.join(eval_output_dir, 'results.pickle')))
    with open(os.path.join(eval_output_dir, 'results.pickle'), 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    hyperparams = get_hyperparams(args=args)
    print('saving hyperparams to {} ....'.format(os.path.join(eval_output_dir, 'hyperparams.pickle')))
    with open(os.path.join(eval_output_dir, 'hyperparams.pickle'), 'wb') as handle:
        pickle.dump(hyperparams, handle, protocol=pickle.HIGHEST_PROTOCOL)

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



def get_results(labels, preds, binary=False):
    if binary:
        accuracy = accuracy_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds)
        precision = precision_score(y_true=labels, y_pred=preds)
        recall = recall_score(y_true=labels, y_pred=preds)
        return {
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc_and_f1": (accuracy + f1) / 2,
        }

    else:
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

def get_accuracy(labels, preds):
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    return {
        "acc": accuracy
    }

def get_hyperparams(args):
    return {
        "bert_model": args.bert_model,
        "dev_datasets_ids": args.dev_datasets_ids,
        "dev_dataset_finetune": args.dev_dataset_finetune,
        "test_dataset_eval": args.test_dataset_eval,
        "max_seq_length": args.max_seq_length,
        "per_gpu_train_batch_size": args.per_gpu_train_batch_size,
        "per_gpu_eval_batch_size": args.per_gpu_eval_batch_size,
        "meta_learn_iter": args.meta_learn_iter,
        "inner_train_steps": args.inner_train_steps
    }


def setup(rank, world_size):
    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="../../data/XNLI-1.0", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    # parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    # parser.add_argument("--bert_model", default="bert-base-multilingual-cased", type=str)
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased", type=str)
    parser.add_argument("--dev_datasets_ids", type=str, default="VDC_ar,corp_PRST_ar_VDC")
    parser.add_argument("--dev_dataset_finetune", type=str, default="corp_PRST_ar_VDC")
    parser.add_argument("--test_dataset_eval", type=str, default="corp_SSM_ar_VDC")
    parser.add_argument("--labels", type=str, default="Main_Consequence,Distant_Evaluation,Cause_Specific,Distant_Anecdotal,Distant_Expectations_Consequences,Main,Distant_Historical,Cause_General")

    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: ")

    parser.add_argument("--model_name_or_path", default='', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list:")

    parser.add_argument("--task_name", default='dp', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))

    parser.add_argument("--cache_dir", default='results/', type=str,
                        help="The directory where the pre-train model is .")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")


    parser.add_argument("--num_classes", default=3, type=int, help="number of classes")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
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
    parser.add_argument("--num_train_epochs", default=3, type=float,
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


    parser.add_argument("--do_finetuning", type=int, default=1, help="whether to apply fine tuning after meta training (few shot) or not (zero shot)."
                                                                      "1 if True, 0 if False")
    parser.add_argument("--do_validation", type=int, default=1, help="whether to validate after every iteration (outerloop) on the validation dataset especially if applying hyperparameter search")

    parser.add_argument("--do_evaluation", type=int, default=1, help="whether to test the trained model on the testing dataset. 1 if True, 0 if False")

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

    parser.add_argument('--meta_learn_iter', default=1, type=int,
                        help='Number of iteration to perform on meta-learning')

    parser.add_argument('--n', default=8, type=int, help='Support samples per class for few-shot tasks')

    parser.add_argument('--inner_train_steps', default=3, type=int,
                        help='Number of inner-loop updates to perform on training tasks')
    parser.add_argument('--inner_lr', default=1e-4, type=float)

    parser.add_argument("--output_dir_meta", default='results/meta/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir_fine_tune", default='results/task_fine_tune/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_task_dir", default='results/task_fine_tune/eval/', type=str,
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

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        pass
        # torch.cuda.set_device(args.local_rank)
        # device = torch.device("cuda", args.local_rank)
        # torch.distributed.init_process_group(backend='nccl')
        # args.n_gpu = 1

    # args.device = device

    from socket import gethostname
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    # gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    gpus_per_node = 1
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)


    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    print('rank: {}, local rank: {}'.format(rank, local_rank))
    torch.cuda.set_device(local_rank)

    args.local_rank = rank
    args.device = torch.device("cuda", args.local_rank)
    device = args.device

    print('device is: {}'.format(args.device))

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

    processor = processors[args.task_name](labels=args.labels)

    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    if 'xlm' in args.bert_model:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels).to(args.device)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels).to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    # get list of datasets for xmaml - meta training
    dev_ids = args.dev_datasets_ids.strip().split(",")
    print('dev ids: {}'.format(dev_ids))
    domains_to_maml = {}
    for id in dev_ids:
        df_path = datasets[id]
        tmp_examples = processor.get_examples(df_path=df_path, df_id=id)
        features = convert_examples_to_features(examples=tmp_examples, label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                tokenizer=tokenizer,
                                                output_mode="classification")
        tmp_dataset = load_cache_examples(features=features)

        domains_to_maml[id] = tmp_dataset

    # get dev dataset for fine tuning
    id_dev_dataset_finetune = args.dev_dataset_finetune
    df_path = datasets[id_dev_dataset_finetune]
    tmp_examples = processor.get_examples(df_path=df_path,  df_id=id)
    features = convert_examples_to_features(examples=tmp_examples, label_list=label_list,
                                            max_seq_length=args.max_seq_length,
                                            tokenizer=tokenizer,
                                            output_mode="classification")
    dev_dataset_finetune = load_cache_examples(features=features)

    # get test dataset for testing (after fine tuning (few shot) or after meta-training (zero-shot))
    id_test_dataset_eval = args.test_dataset_eval
    df_path = datasets[id_test_dataset_eval]
    tmp_examples = processor.get_examples(df_path=df_path,  df_id=id)
    features = convert_examples_to_features(examples=tmp_examples, label_list=label_list,
                                            max_seq_length=args.max_seq_length,
                                            tokenizer=tokenizer,
                                            output_mode="classification")
    test_dataset_eval = load_cache_examples(features=features)

    logger.info("Training/evaluation parameters %s", args)

    logger.info("Running XMAML ...")
    # examples, label_list, max_seq_length, tokenizer, output_mode
    if args.do_validation:
        x_maml_with_aux_langs(args=args, model=model, lang_datasets=domains_to_maml, validation_dataset=dev_dataset_finetune, tokenizer=tokenizer, suffix=None)
    else:
        x_maml_with_aux_langs(args=args, model=model, lang_datasets=domains_to_maml, tokenizer=tokenizer, suffix=None)

    if args.do_finetuning == 1:
        logger.info("Fine Tuning ...")
        # Fine tuning
        model = BertForSequenceClassification.from_pretrained(args.output_dir_meta, num_labels=num_labels)
        model = model.to(args.device)

        fine_tune_task(args, model=model, train_dataset=dev_dataset_finetune, tokenizer=tokenizer)

    torch.distributed.destroy_process_group()

    logger.info("Evaluating ...")

    if args.do_finetuning == 1:
        if args.do_evaluation == 1:
            # model loaded already
            evaluate_task(args, model=model, eval_dataset=test_dataset_eval, label_list=label_list, prefix="")
        else:
            logger.info("Will not apply evaluation as --do_evaluation=0")
    else:
        if args.do_evaluation == 1:
            # Load the model then evaluate
            model = BertForSequenceClassification.from_pretrained(args.output_dir_meta, num_labels=num_labels)
            model = model.to(args.device)
            evaluate_task(args, model=model, eval_dataset=test_dataset_eval, label_list=label_list, prefix="")
        else:
            logger.info("Will not apply evaluation as --do_evaluation=0")


if __name__ == "__main__":
    main()
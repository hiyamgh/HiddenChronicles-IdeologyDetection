import os
import pickle

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import BertModel, BertTokenizer
from arabert import ArabertPreprocessor


def get_context(token_ids, target_position, sequence_length=128):
    """
    Given a text containing a target word, return the sentence snippet which surrounds the target word
    (and the target word's position in the snippet).

    :param token_ids: list of token ids (for an entire line of text)
    :param target_position: index of the target word's position in `tokens`
    :param sequence_length: desired length for output sequence (e.g. 128, 256, 512)
    :return: (context_ids, new_target_position)
                context_ids: list of token ids for the output sequence
                new_target_position: index of the target word's position in `context_ids`
    """
    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 2) / 2)
    context_start = max([0, target_position - window_size])
    padding_offset = max([0, window_size - target_position])
    padding_offset += max([0, target_position + window_size - len(token_ids)])

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += padding_offset * [0]

    new_target_position = target_position - context_start

    return context_ids, new_target_position


def collect_from_coha(target_words,
                      years,
                      sequence_length,
                      pretrained_weights='models/bert-base-uncased',
                      df_path='../bert_predictions/nahar/',
                      text_column='Sentence',
                      prediction_column='Prediction',
                      output_path=None,
                      buffer_size=1024):
    """
    Collect usages of target words from the COHA dataset.

    :param target_words: list of words whose usages are to be collected
    :param decades: list of year integers (e.g. list(np.arange(1910, 2001, 10)))
    :param sequence_length: the number of tokens in the context of a word occurrence
    :param pretrained_weights: path to model folder with weights and config file
    :param coha_dir: path to COHA directory (containing `all_1810.txt`, ..., `all_2000.txt`)
    :param output_path: path to output file for `usages` dictionary. If provided, data is stored
                        in this file incrementally (use e.g. to avoid out of memory errors)
    :param buffer_size: (max) number of usages to process in a single model run
    :return: usages: a dictionary from target words to lists of usage tuples
                     lemma -> [(vector, sentence, word_position, decade), (v, s, p, d), ...]
    """

    # load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertModel.from_pretrained(pretrained_weights)
    arabert_prep = ArabertPreprocessor(model_name=pretrained_weights)
    if torch.cuda.is_available():
        model.to('cuda')

    # build word-index vocabulary for target words
    i2w = {}
    for t, t_id in zip(target_words, tokenizer.encode(' '.join(target_words))):
        i2w[t_id] = t

    # buffers for batch processing
    batch_input_ids = []
    batch_tokens = []
    batch_pos = []
    batch_snippets = []
    batch_decades = []
    batch_predictions = []

    usages = defaultdict(list)  # w -> (vector, sentence, word_position, decade)

    # do collection
    for T, year in enumerate(years):
        # one time interval at a time
        print('Year {}...'.format(year))
        df = pd.read_excel(os.path.join(df_path, 'df_test_{}.xlsx'.format(year)))
        df = df.sample(n=2000, random_state=42)
        print('len of df: {}'.format(len(df)))
        # with open('{}/all_{}.txt'.format(coha_dir, decade), 'r') as f:
        #     lines = f.readlines()
        lines = list(df[text_column])
        predictions = list(df[prediction_column])

        for L, line in enumerate(lines):

            tokens_words = str(line).strip().split(' ')
            # Apply AraBERT preprocessing function before doing any encoding
            line = arabert_prep.preprocess(line)
            pred_label = predictions[L]
            # tokenize line and convert to token ids
            tokens = tokenizer.encode(line)

            for pos, token in enumerate(tokens):
                if token in i2w:
                # if (tokens_words[pos] in target_words) or any([tokens_words[pos] in t for t in target_words]):
                    # store usage info of target words only   .. here we account for OCR errors (maybe a target word is part of a word but no space is found
                    # if token in i2w or any([t in i2w[token] for t in target_words]):

                    # if tokens_words[pos] in target_words:
                    #     corr_token = token
                    # else:
                    #     for t in target_words:
                    #         if tokens_words[pos] in t:
                    #             corr_token = t
                    #             break

                    context_ids, pos_in_context = get_context(tokens, pos, sequence_length)
                    input_ids = [101] + context_ids + [102]

                    # convert later to save storage space
                    # snippet = tokenizer.convert_ids_to_tokens(context_ids)

                    # add usage info to buffers
                    batch_input_ids.append(input_ids)
                    batch_tokens.append(i2w[token])
                    # batch_tokens.append(corr_token)
                    batch_pos.append(pos_in_context)
                    batch_snippets.append(context_ids)
                    batch_decades.append(year)
                    batch_predictions.append(pred_label)

                    # if the buffers are full...             or if we're at the end of the dataset
                if (len(batch_input_ids) >= buffer_size) or (L == len(lines) - 1 and T == len(years) - 1):

                    with torch.no_grad():
                        # collect list of input ids into a single batch tensor
                        input_ids_tensor = torch.tensor(batch_input_ids)
                        if torch.cuda.is_available():
                            input_ids_tensor = input_ids_tensor.to('cuda')

                        # run usages through language model
                        outputs = model(input_ids_tensor, output_hidden_states=True)
                        if torch.cuda.is_available():
                            hidden_states = [l.detach().cpu().clone().numpy() for l in outputs[2]]
                        else:
                            hidden_states = [l.clone().numpy() for l in outputs[2]]

                        # get usage vectors from hidden states
                        hidden_states = np.stack(hidden_states)  # (13, B, |s|, 768)
                        # print('Expected hidden states size: (13, B, |s|, 768). Got {}'.format(hidden_states.shape))
                        # usage_vectors = np.sum(hidden_states, 0)  # (B, |s|, 768)
                        # usage_vectors = hidden_states.view(hidden_states.shape[1],
                        #                                    hidden_states.shape[2],
                        #                                    -1)
                        usage_vectors = np.sum(hidden_states[1:, :, :, :], axis=0)
                        # usage_vectors = hidden_states.reshape((hidden_states.shape[1], hidden_states.shape[2], -1))

                    if output_path and os.path.exists(output_path):
                        with open(output_path, 'rb') as f:
                            usages = pickle.load(f)

                    # store usage tuples in a dictionary: lemma -> (vector, snippet, position, decade)
                    for b in np.arange(len(batch_input_ids)):
                        usage_vector = usage_vectors[b, batch_pos[b] + 1, :]
                        usages[batch_tokens[b]].append(
                            (usage_vector, batch_snippets[b], batch_pos[b], batch_decades[b], batch_predictions[b]))

                    print(set(batch_tokens))

                    # finally, empty the batch buffers
                    batch_input_ids, batch_tokens, batch_pos, batch_snippets, batch_decades, batch_predictions = [], [], [], [], [], []

                    # and store data incrementally
                    if output_path:
                        with open(output_path, 'wb') as f:
                            pickle.dump(usages, file=f)

    return usages


if __name__ == '__main__':
    collect_from_coha(target_words=['مقاومه', 'اسرائيل', 'سوري'],
                      years=[1982, 1985, 1986],
                      sequence_length=128,
                      pretrained_weights='aubmindlab/bert-base-arabertv2',
                      # coha_dir='data/coha',
                      df_path='../bert_predictions/nahar/',
                      text_column='Sentence',
                      prediction_column='Prediction',
                      output_path='usages_nahar_gpu.pkl',
                      buffer_size=1024)
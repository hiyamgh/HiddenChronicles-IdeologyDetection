import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pickle
import gc
import re
import pandas as pd
import argparse
from tqdm import tqdm


def remove_url(text, replace_token):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_shifts(input_path):
    shifts_dict = {}
    df_shifts = pd.read_csv(input_path, sep=',', encoding='utf8')
    for idx, row in df_shifts.iterrows():
        shifts_dict[row['word']] = row['shift_index']
    return shifts_dict


def tokens_to_batches_specific(ds, tokenizer, batch_size, max_length, words):
    batches = []
    batch = []
    batch_counter = 0

    print('Dataset: ', ds)
    counter = 0
    with open(ds, 'r', encoding='utf8') as fout:
        num = len(fout.readlines())
    fout.close()


    with open(ds, 'r', encoding='utf8') as f:

        for line in f:
            counter += 1

            line = line.strip()

            found = False
            for w in words:
                if (w in line.strip()) or (any([w in t for t in line.split()])):
                    found = True

            if found:

                if len(line.split()) > max_length - 2:
                    sents = []
                    line = line.split()
                    for i in range(0, len(line), max_length - 2):
                        sent = line[i: i + max_length - 2]
                        marked_sent = "[CLS] " + " ".join(sent) + " [SEP]"
                        sents.append(marked_sent)
                else:
                    sents = ["[CLS] " + line + " [SEP]"]

                marked_text = " ".join(sents)

                tokenized_text = tokenizer.tokenize(marked_text)

                for i in range(0, len(tokenized_text), max_length):

                    batch_counter += 1
                    input_sequence = tokenized_text[i:i + max_length]

                    indexed_tokens = tokenizer.convert_tokens_to_ids(input_sequence)

                    batch.append((indexed_tokens, input_sequence))

                    if batch_counter % batch_size == 0:
                        batches.append(batch)
                        batch = []

            if counter % 100 == 0:
                print('count: {}/{}'.format(counter, num))
    print()
    print('Tokenization done!')
    print('len batches: ', len(batches))

    return batches


def tokens_to_batches2(ds, tokenizer, batch_size, max_length):
    batches = []
    batch = []
    batch_counter = 0

    print('Dataset: ', ds)
    counter = 0

    with open(ds, 'r', encoding='utf8') as fout:
        num = len(fout.readlines())
    fout.close()

    with open(ds, 'r', encoding='utf8') as f:

        for line in f:
            counter += 1

            line = line.strip()

            if len(line.split()) > max_length - 2:
                sents = []
                line = line.split()
                for i in range(0, len(line), max_length - 2):
                    sent = line[i: i + max_length - 2]
                    marked_sent = "[CLS] " + " ".join(sent) + " [SEP]"
                    sents.append(marked_sent)
            else:
                sents = ["[CLS] " + line + " [SEP]"]

            marked_text = " ".join(sents)

            tokenized_text = tokenizer.tokenize(marked_text)

            for i in range(0, len(tokenized_text), max_length):

                batch_counter += 1
                input_sequence = tokenized_text[i:i + max_length]

                indexed_tokens = tokenizer.convert_tokens_to_ids(input_sequence)

                batch.append((indexed_tokens, input_sequence))

                if batch_counter % batch_size == 0:
                    batches.append(batch)
                    batch = []

            if counter == 40:
                break
            if counter % 100 == 0:
                print('count: {}/{}'.format(counter, num))
    print()
    print('Tokenization done!')
    print('len batches: ', len(batches))

    return batches

def get_token_embeddings(batches, model, batch_size):

    token_embeddings = []
    tokenized_text = []
    counter = 0

    for batch in tqdm(batches):
        counter += 1
        if counter % 1000 == 0:
            print('Generating embedding for batch: ', counter)
        lens = [len(x[0]) for x in batch]
        max_len = max(lens)
        tokens_tensor = torch.zeros(batch_size, max_len, dtype=torch.long).to(device)
        segments_tensors = torch.ones(batch_size, max_len, dtype=torch.long).to(device)
        batch_idx = [x[0] for x in batch]
        batch_tokens = [x[1] for x in batch]

        for i in range(batch_size):
            length = len(batch_idx[i])

            for j in range(max_len):
                if j < length:
                    tokens_tensor[i][j] = batch_idx[i][j]

        # Predict hidden states features for each layer
        with torch.no_grad():
            model_output = model(tokens_tensor, segments_tensors)
            encoded_layers = model_output[-1][-4:] #last four layers of the encoder


        words = ['']
        for batch_i in range(batch_size):

            # For each token in the sentence...
            for token_i in range(len(batch_tokens[batch_i])):


                # Holds last 4 layers of hidden states for each token
                hidden_layers = []

                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]

                    hidden_layers.append(vec)

                hidden_layers = torch.sum(torch.stack(hidden_layers)[-4:], 0).reshape(1, -1).detach().cpu().numpy()

                token_embeddings.append(hidden_layers)
                tokenized_text.append(batch_tokens[batch_i][token_i])

    return token_embeddings, tokenized_text


def average_save_and_print(vocab_vectors, save_path):
    for k, v in vocab_vectors.items():

        if len(v) == 2:
            avg = v[0] / v[1]
            vocab_vectors[k] = avg

    with open(save_path, 'wb') as handle:
        pickle.dump(vocab_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)




def get_time_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length):

    vocab_vectors = {}
    vocab_vectors_avg = {}

    for ds in datasets:

        year = ds[-8:-4]

        # all_batches = tokens_to_batches(ds, tokenizer, batch_size, max_length)
        # all_batches = tokens_to_batches2(ds, tokenizer, batch_size, max_length)
        all_batches = tokens_to_batches_specific(ds, tokenizer, batch_size, max_length, ['اسرائيل'])
        chunked_batches = chunks(all_batches, 1000)
        num_chunk = 0

        for batches in chunked_batches:
            num_chunk += 1
            print('Chunk ', num_chunk)

            token_embeddings, tokenized_text = get_token_embeddings(batches, model, batch_size)

            splitted_tokens = []
            splitted_array = np.zeros((1, 768))
            prev_token = ""
            prev_array = np.zeros((1, 768))

            for i, token_i in enumerate(tokenized_text):

                array = token_embeddings[i]

                if token_i.startswith('##'):

                    if prev_token:
                        splitted_tokens.append(prev_token)
                        prev_token = ""
                        splitted_array = prev_array

                    splitted_tokens.append(token_i)
                    splitted_array += array

                else:

                    if token_i + '_' + year in vocab_vectors:
                        vocab_vectors[token_i + '_' + year][0] += array
                        vocab_vectors[token_i + '_' + year][1] += 1
                    else:
                        vocab_vectors[token_i + '_' + year] = [array, 1]

                    if splitted_tokens:
                        sarray = splitted_array / len(splitted_tokens)
                        stoken_i = "".join(splitted_tokens).replace('##', '')


                        if stoken_i + '_' + year in vocab_vectors:
                            vocab_vectors[stoken_i + '_' + year][0] += sarray
                            vocab_vectors[stoken_i + '_' + year][1] += 1
                        else:
                            vocab_vectors[stoken_i + '_' + year] = [sarray, 1]

                        splitted_tokens = []
                        splitted_array = np.zeros((1, 768))

                    prev_array = array
                    prev_token = token_i

            del token_embeddings
            del tokenized_text
            del batches
            gc.collect()

        print('Sentence embeddings generated.')

    print("Length of vocab after training: ", len(vocab_vectors.items()))

    average_save_and_print(vocab_vectors, embeddings_path)
    del vocab_vectors
    del vocab_vectors_avg
    gc.collect()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument('--bert_model', type=str, help='Path to initial pretrained model',
                        default='aubmindlab/bert-base-arabertv2')

    parser.add_argument('--path_to_model', type=str,
                        help='Paths to the fine-tuned BERT model',
                        # default='/onyx/data/p118/models_ft_ar/checkpoint-999000/pytorch_model.bin')
                        # default='E:/checkpoint-99900/pytorch_model.bin')
                        default='aubmindlab/bert-base-arabertv2')

    parser.add_argument('--path_to_datasets', type=str,
                        help='Paths to each of the time period specific corpus separated by ;',
                        # default='/onyx/data/p118/data/1982.txt;/onyx/data/p118/data/1983.txt;/onyx/data/p118/data/1984.txt;/onyx/data/p118/data/1985.txt;/onyx/data/p118/data/1986.txt')
                        # default='E:/nahar/1982.txt;E:/nahar/1983.txt;E:/nahar/1984.txt;E:/nahar/1985.txt;E:/nahar/1986.txt')
                        default='C:/Users/96171/Downloads/LiverpoolFC_2013.txt')
                        # default='C:/Users/96171/Downloads/1982.txt',)

    parser.add_argument('--embeddings_path', type=str,
                        help='Path to output time embeddings',
                        default='embeddings/nahar-1982-1986.pickle')

    args = parser.parse_args()

    datasets = args.path_to_datasets.split(';')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    if args.bert_model == args.path_to_model:
        model = BertModel.from_pretrained(args.bert_model, output_hidden_states=True)
    else:
        if torch.cuda.is_available():
            state_dict = torch.load(args.path_to_model)
        else:
            state_dict = torch.load(args.path_to_model, map_location=torch.device('cpu'))

        model = BertModel.from_pretrained(args.bert_model, state_dict=state_dict, output_hidden_states=True)

    # model.cuda()
    model.to(device)
    model.eval()

    embeddings_path = args.embeddings_path

    get_time_embeddings(embeddings_path, datasets, tokenizer, model, args.batch_size, args.max_length)

    #Pearson coefficient:  (0.4680305011683137, 1.3380286367013385e-06) 84036

    # /onyx/data/p118/data/1982.txt
    # /onyx/data/p118/data/1983.txt
    # /onyx/data/p118/data/1984.txt
    # /onyx/data/p118/data/1985.txt
    # /onyx/data/p118/data/1986.txt
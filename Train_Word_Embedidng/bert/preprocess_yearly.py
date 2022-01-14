from preprocess import ArabertPreprocessor
import time
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--dir", default='/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/data/nahar/', help="prefix for embedding for split a")
parser.add_argument("--out_dir", default='/scratch/7613491_hkg02/political_discourse_mining_hiyam/arabert/nahar-preprocessed/', help="prefix for embedding for split a")
parser.add_argument("--file", default='2006.txt', help="prefix for embedding for split b")
args = parser.parse_args()

model_name = "aubmindlab/bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

t1 = time.time()
with open(os.path.join(args.out_dir, args.file), 'w', encoding='utf-8') as f1:

    with open(os.path.join(args.dir, args.file), 'r', encoding='utf-8') as f2:
        all_text = f2.readlines()
        for ln in all_text:
            ll_prep = arabert_prep.preprocess(ln)
            f1.write(ll_prep + '\n')
    f2.close()
f1.close()
t2 = time.time()

print('time taken: {} mins'.format((t2-t1)/60))

with open(os.path.join(args.dir, args.file), 'r', encoding='utf-8') as f:
    old_text = f.readlines()
f.close()

with open(os.path.join(args.out_dir, args.file), 'r', encoding='utf-8') as f:
    new_text = f.readlines()
f.close()

print(len(old_text))
print(len(new_text))
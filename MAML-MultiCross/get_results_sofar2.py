import os
import pandas as pd
import pickle

# df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1'])
df = pd.DataFrame(columns=['val_accuracy_mean', 'train_accuracy_mean', 'epoch', 'model'])
# /onyx/data/p118/MAML-MultiCross/all_experiments-threewayprotomaml/
for subdir, dirs, files in os.walk('all_langs-threewayprotomaml/'):
    for file in files:
        val_acc, train_acc = [], []
        if file == 'summary_statistics.csv':
            # with open(os.path.join(subdir, file), 'rb') as handle:
            #     result = pickle.load(handle)
            # with open(os.path.join(subdir, 'hyperparams.pickle'), 'rb') as handle:
            #     hyperparams = pickle.load(handle)
            # print(os.path.join(subdir, file))

            summ = pd.read_csv(os.path.join(subdir, file))
            print(subdir)
            for i, row in summ.iterrows():
                val_acc.append(row['val_accuracy_mean'])
                train_acc.append(row['train_accuracy_mean'])
            model = subdir
            # for k in result:
            #     print('{}: {}'.format(k, result[k]))
            # print('model: {}'.format(hyperparams['bert_model']))
            # print('dev: {}'.format(hyperparams['dev_datasets_ids']))
            df = df.append({
                'val_accuracy_mean': val_acc[-1],
                'train_accuracy_mean': train_acc[-1],
                'epoch': list(summ['epoch'])[-1],
                'model': model
            }, ignore_index=True)
            print('==============================================================')

df = df.sort_values(by=['val_accuracy_mean'], ascending=False)
print(df)
df.to_csv('results2.csv', index=False)
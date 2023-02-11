import os
import pickle
import numpy as np


if __name__ == '__main__':

    rootdir = 'results_tuning/'
    results = []
    for subdir, dirs, files in os.walk(rootdir):
        if 'validation_losses.pkl' in files:
            with open(os.path.join(subdir, 'validation_losses.pkl'), 'rb') as handle:
                validation_losses = np.array(pickle.load(handle))

            with open(os.path.join(subdir, 'validation_accuracies.pkl'), 'rb') as handle:
                validation_accs = np.array(pickle.load(handle))

            avg_loss = validation_losses[~np.isnan(validation_losses)].mean()
            avg_acc = validation_accs[~np.isnan(validation_accs)].mean()
            print('Avg loss: {}, Avg accuracy: {}, model: {}'.format(avg_loss, avg_acc, subdir))
            results.append((avg_loss, avg_acc, subdir))

    results_sorted = list(sorted(results, key=lambda tup: tup[0]))
    for r in results_sorted:
        print(r)
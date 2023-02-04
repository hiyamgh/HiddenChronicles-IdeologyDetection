import os
import pickle
import numpy as np

if __name__ == '__main__':

    rootdir = 'results_tuning/'

    for subdir, dirs, files in os.walk(rootdir):
        if 'validation_losses.pkl' in files:
            with open(os.path.join(subdir, 'validation_losses.pkl'), 'rb') as handle:
                validation_losses = pickle.load(handle)

            with open(os.path.join(subdir, 'validation_accuracies.pkl'), 'rb') as handle:
                validation_accs = pickle.load(handle)

            print('Avg loss: {}, Avg accuracy: {}, model: {}'.format(np.mean(validation_losses), np.mean(validation_accs), subdir))
import pickle
import os

for year in range(1934, 2006):
    with open('D:/results_deltas/{}-disps.pkl'.format(year), 'rb') as handlle:
        result = pickle.load(handlle)

    print('{}: {}'.format(year, result))
import pandas as pd
import os

directory = "translations_joined_cleaned/"
for file in os.listdir(directory):
    df = pd.read_excel(os.path.join(directory, file))
    labels = list(set(df['Label']))

    for i, label in enumerate(labels):
        if i != len(labels) - 1:
            print("{},".format(label), end="")
        else:
            print("{}".format(label), end="")
    break

print('\n')
for file in os.listdir(directory):
    df = pd.read_excel(os.path.join(directory, file))
    labels = list(set(df['Speech_label']))

    for i, label in enumerate(labels):
        if i != len(labels) - 1:
            print("{},".format(label), end="")
        else:
            print("{}".format(label), end="")
    break
import pandas as pd
import os

dirs_ARG = [
    'translationsv2/argumentation/Palestinian Resistance South Lebanon/',
    'translationsv2/argumentation/Sabra and Shatila Massacre/'
]

dirs_propaganda = [
    'translationsv2/propaganda/Palestinian Resistance South Lebanon/',
    'translationsv2/propaganda/Sabra and Shatila Massacre/'
]

dirs_van_dijk_contexts = [
    'translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/',
    'translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/'
]

dirs_van_dijk_speeches = [
    'translationsv2/van_dijk_speeches/Palestinian Resistance South Lebanon/',
    'translationsv2/van_dijk_speeches/Sabra and Shatila Massacre/'
]

labels = set()
for dir in dirs_ARG:
    for file in os.listdir(dir):
        df = pd.read_excel(os.path.join(dir, file))
        labels_df = list(df['label'])
        for lbl in labels_df:
            labels.add(lbl)

print('ARG labels:')
for i, lbl in enumerate(list(labels)):
    if i != len(labels) - 1:
        print('{},'.format(lbl), end="")
    else:
        print('{}'.format(lbl), end="")
print('\n\n')

##################################################
labels = set()
for dir in dirs_propaganda:
    for file in os.listdir(dir):
        df = pd.read_excel(os.path.join(dir, file))
        labels_df = list(df['label'])
        for lbl in labels_df:
            labels.add(lbl)

print('Propaganda labels:')
for i, lbl in enumerate(list(labels)):
    if i != len(labels) - 1:
        print('{};'.format(lbl), end="")
    else:
        print('{}'.format(lbl), end="")
print('\n\n')
##############################################################

labels = set()
for dir in dirs_van_dijk_contexts:
    for file in os.listdir(dir):
        df = pd.read_excel(os.path.join(dir, file))
        labels_df = list(df['label'])
        for lbl in labels_df:
            labels.add(lbl)

print('Van Dijk Contexts labels:')
for i, lbl in enumerate(list(labels)):
    if i != len(labels) - 1:
        print('{},'.format(lbl), end="")
    else:
        print('{}'.format(lbl), end="")
print('\n\n')


##############################################################

labels = set()
for dir in dirs_van_dijk_speeches:
    for file in os.listdir(dir):
        df = pd.read_excel(os.path.join(dir, file))
        labels_df = list(df['label'])
        for lbl in labels_df:
            labels.add(lbl)

print('Van Dijk Speeches labels:')
for i, lbl in enumerate(list(labels)):
    if i != len(labels) - 1:
        print('{},'.format(lbl), end="")
    else:
        print('{}'.format(lbl), end="")
print('\n\n')
'''
Hiyam - doing this code in order to extract annotations and save them as a csv file for
later usage
'''
import os
import pandas as pd

# start with one of the labels
# have a unit at the end

# start with continued
# until you reach a label -- this will be the label of the sentence
# might end with a no-unit.


# cases:
# 1. is a title
# 2. is a par-sep
# 3. is one of the labels, preceeded by par-sep
# 5. is one of the labels, preceeded by continued
# is one of the labels, preceeded by no-unit
# is a no-unit preceeded by label
# is a continued preceeded by no-unit or par-sep
# is a continued preceeded by continued
#

path1 = 'annotated-txt/split-by-portal-final/aljazeera/'
path2 = 'annotated-txt/split-by-portal-final/foxnews/'
path3 = 'annotated-txt/split-by-portal-final/guardian/'
paths = [path1, path2, path3]

df = pd.DataFrame(columns=['Sentence', 'Label'])

for path in paths:
    editorials = os.listdir(path)

    sentences = {}
    last_unit_nb = 0
    last_unit_lbl = ''
    for editorial in editorials:
        with open(os.path.join(path, editorial), 'r') as f:
            lines = f.readlines()

            # get the index of 'no-unit .' (whenever we have stop mark then this marks a complete sentence)
            idxs = []
            for i, line in enumerate(lines):
                # _, unit_lbl, text = line.split('\t')
                split_line = line.split('\t')
                unit_lbl = split_line[1]
                text = split_line[2:]
                text = ''.join(text)
                if unit_lbl.strip() == 'no-unit' and text.strip() == '.':
                    idxs.append(i)
            chunks = []
            for j, idx in enumerate(idxs):
                if idx == idxs[0]: # first index
                    chunks.append(lines[0: idxs[j]+1])
                else:
                    if j != len(idxs) - 1:
                        chunks.append(lines[idxs[j-1]+1: idxs[j]+1])
                    else:
                        chunks.append(lines[idxs[j-1]+1:])

            sentences = []
            # transform each chunk into a sentence
            for chunk in chunks:
                sentence = ''
                actual_label = 'no-unit'
                for line in chunk:
                    split_line = line.split('\t')

                    label = split_line[1]

                    text = ' '.join(split_line[2:])
                    text = text.replace('\n', '')
                    text = text.replace('\t', ' ')

                    if label == 'title':
                        continue
                    if label == 'par-sep':
                        continue

                    if label in ['continued', 'no-unit']:
                        pass
                    else:
                        actual_label = label

                    sentence += ' ' + text

                sentences.append((sentence.strip(), actual_label))
                df = df.append({
                    'Sentence': sentence.strip(),
                    'Label': actual_label
                }, ignore_index=True)

            # print()
df = df.to_excel('sentences_annotations.xlsx', index=False)
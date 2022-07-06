import os
import pandas as pd


gold_label_files = ['dev-task-flc-tc.labels', 'train-task-flc-tc.labels']
article_dirs = ['dev-articles/', 'train-articles/']
dataset_names = ['dev-articles.csv', 'train-articles.csv']
labels_political_discourse = ['active', 'euphimism', 'details', 'exaggeration', 'bragging', 'litote', 'repetition',
                                  'metaphor', 'he said', 'apparent denial', 'apparent concession', 'blame transfer',
                                  'other kinds', 'opinion', 'irony']

for i, article_dir in enumerate(article_dirs):
    df = pd.DataFrame(columns=['Sentence', 'Technique', 'Span', 'article_id'])
    sentences, techniques, spans, article_ids = [], [], [], []
    label_file = gold_label_files[i]
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            article_id, technique, begin_offset, end_offset = line.split('\t')
            end_offset = int(end_offset[:-1])
            begin_offset = int(begin_offset)

            with open(os.path.join(article_dir, 'article{}.txt'.format(article_id)), 'r', encoding='utf-8') as fa:
                article_str = fa.read()
                span = article_str[begin_offset:end_offset]

                begin_idx = 0 # if begin_offset is 0
                # go backwards from begin_offset until you find a \n, this is the start of the sentence that contains the span
                for j in range(begin_offset, 0, -1):
                    if article_str[j] == '\n':
                        begin_idx = j
                        break

                end_idx = len(article_str) # if end_offset is len(article_str)
                # go forward from end_offset until you find a \n, this is the end of the sentence that contains the span
                for j in range(end_offset, len(article_str)):
                    if article_str[j] == '\n':
                        end_idx = j
                        break

                if article_str[begin_idx] == '\n':
                    sentence = article_str[begin_idx+1:end_idx]
                else:
                    sentence = article_str[begin_idx:end_idx]

                sentences.append(sentence)
                techniques.append(technique)
                spans.append(span)
                article_ids.append(article_id)

    df['Sentence'] = sentences
    df['Technique'] = techniques
    df['Span'] = spans
    df['article_id'] = article_ids

    df = df.sort_values(by='Sentence')

    for label in labels_political_discourse:
        df[label] = ""

    df.to_csv('{}'.format(dataset_names[i]), index=False)
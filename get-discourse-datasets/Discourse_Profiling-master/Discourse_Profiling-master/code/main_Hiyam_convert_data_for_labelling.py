import os
from process_file import process_doc
import pandas as pd

train_data = []
validate_data = []
test_data = []
df = pd.DataFrame(columns=['Sentence', 'Label', 'Domain'])
count = 0
sentences, labels, domains = [], [], []
for domain in ["Business", "Politics", "Crime", "Disaster", "kbp"]:
    subdir = "../data/train/" + domain
    files = os.listdir(subdir)
    for file in files:
        if '.txt' in file:
            doc = process_doc(os.path.join(subdir, file), domain)  # '../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
            # print(doc.sent_to_event)
            train_data.append(doc)

    subdir = "../data/test/" + domain
    files = os.listdir(subdir)
    for file in files:
        if '.txt' in file:
            doc = process_doc(os.path.join(subdir, file),  domain)  # '../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
            # print(doc.sent_to_event)
            test_data.append(doc)

    subdir = "../data/validation"
    files = os.listdir(subdir)
    for file in files:
        if '.txt' in file:
            doc = process_doc(os.path.join(subdir, file), 'VAL') #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
            #print(doc.sent_to_event)
            validate_data.append(doc)
    print(len(train_data), len(validate_data), len(test_data), len(train_data) + len(validate_data) + len(test_data))

    # test_data: list of Documents
    # each Document contains:
    # sentences: OrderedDict, key is S1, value is list of words
    # sent_to_event: sentence id as key, label as value
    for doc in train_data:
        for sentence in doc.sentences:
            sentences.append(' '.join(doc.sentences[sentence]))
            labels.append(doc.sent_to_event[sentence])
            domains.append(domain)

            count += 1

    for doc in test_data:
        for sentence in doc.sentences:
            sentences.append(' '.join(doc.sentences[sentence]))
            labels.append(doc.sent_to_event[sentence])
            domains.append(domain)

            count += 1

print('total number of sentences: {}'.format(count))
labels_political_discourse = ['active', 'euphimism', 'details', 'exaggeration',	'bragging',	'litote', 'repetition',
                              'metaphor', 'he said', 'apparent denial', 'apparent concession', 'blame transfer',
                              'other kinds', 'opinion', 'irony']

df['Sentence'] = sentences
df['Label'] = labels
df['Domain'] = domains
for label in labels_political_discourse:
    df[label] = ""

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('NewsDiscourse_politicaldiscourse.csv', index=False)
df.to_excel('NewsDiscourse_politicaldiscourse.xlsx', index=False)

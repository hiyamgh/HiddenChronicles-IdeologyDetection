import os
from process_file import process_doc
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser



if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--drop', help='DROP', default=6, type=float)
    # parser.add_argument('--learn_rate', help='LEARNING RATE', default=0, type=float)
    # parser.add_argument('--loss_wt', help='LOSS WEIGHTS', default=0, type=str)
    parser.add_argument('--seed', help='SEED', default=0, type=int)

    args = parser.parse_args()


    SPEECH = 0
    if SPEECH:
        out_map = {'NA':0, 'Speech':1}
    else:
        out_map = {'NA':0,'Main':1,'Main_Consequence':2, 'Cause_Specific':3, 'Cause_General':4, 'Distant_Historical':5,
        'Distant_Anecdotal':6, 'Distant_Evaluation':7, 'Distant_Expectations_Consequences':8}

    train_data = []
    validate_data = []
    test_data = []
    for domain in ["Business", "Politics", "Crime", "Disaster", "kbp"]:
        subdir = "../data/train/"+domain
        files = os.listdir(subdir)
        for file in files:
            if '.txt' in file:
                doc = process_doc(os.path.join(subdir, file), domain) #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
                #print(doc.sent_to_event)
                train_data.append(doc)
        subdir = "../data/test/"+domain
        files = os.listdir(subdir)
        for file in files:
            if '.txt' in file:
                doc = process_doc(os.path.join(subdir, file), domain) #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
                #print(doc.sent_to_event)
                test_data.append(doc)


    subdir = "../data/validation"
    files = os.listdir(subdir)
    for file in files:
        if '.txt' in file:
            doc = process_doc(os.path.join(subdir, file), 'VAL') #'../data/Business/nyt_corpus_data_2007_04_27_1843240.txt'
            #print(doc.sent_to_event)
            validate_data.append(doc)
    print(len(train_data), len(validate_data), len(test_data))

    sum = 0
    import pandas as pd

    labels2gen = {
        'Main': 'main_contents',
        'Main_Consequence': 'main_contents',
        'Cause_Specific': 'context_informing_contents',
        'Cause_General': 'context_informing_contents',
        'Distant_Historical': 'additional_supportive_contents',
        'Distant_Anecdotal': 'additional_supportive_contents',
        'Distant_Evaluation': 'additional_supportive_contents',
        'Distant_Expectations_Consequences': 'additional_supportive_contents',
    }

    # save the training data
    sentences, labels, labels_general = [], [], []
    for doc in train_data:
        for sent_id, sent in doc.sentences.items():
            sent_txt = ' '.join(sent)
            label = doc.sent_to_event[sent_id]
            if label != 'NA':
                sentences.append(sent_txt)
                labels.append(label)
                labels_general.append(labels2gen[label])

    df = pd.DataFrame()
    df['Sentence'] = sentences
    df['Label'] = labels
    df['Label_general'] = labels_general
    df.to_csv('df_train.csv', index=False)
    sum += len(df)

    # save the validation data
    sentences, labels, labels_general = [], [], []
    for doc in validate_data:
        for sent_id, sent in doc.sentences.items():
            sent_txt = ' '.join(sent)
            label = doc.sent_to_event[sent_id]
            if label != 'NA':
                sentences.append(sent_txt)
                labels.append(label)
                labels_general.append(labels2gen[label])
    df = pd.DataFrame()
    df['Sentence'] = sentences
    df['Label'] = labels
    df['Label_general'] = labels_general
    df.to_csv('df_validation.csv', index=False)
    sum += len(df)

    # save the testing data
    sentences, labels, labels_general = [], [], []
    for doc in test_data:
        for sent_id, sent in doc.sentences.items():
            sent_txt = ' '.join(sent)
            label = doc.sent_to_event[sent_id]
            if label != 'NA':
                sentences.append(sent_txt)
                labels.append(label)
                labels_general.append(labels2gen[label])
    df = pd.DataFrame()
    df['Sentence'] = sentences
    df['Label'] = labels
    df['Label_general'] = labels_general
    df.to_csv('df_test.csv', index=False)
    sum += len(df)

    print(sum)

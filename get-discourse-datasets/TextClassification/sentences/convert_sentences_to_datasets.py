import pandas as pd
from googletrans import Translator
import json, time
from tqdm import tqdm


sentences, labels, labels_general = [], [], []
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
with open('sentences.json', encoding="utf-8") as json_file:
    data = json.load(json_file)

    for num in data:
        sentence = data[num]['sentence'].strip()
        label = data[num]['label'].strip()
        if label != 'NAN':
            label_general = labels2gen[label]

            sentences.append(sentence)
            labels.append(label)
            labels_general.append(label_general)

df = pd.DataFrame()
df['Sentence'] = sentences
df['Label'] = labels
df['Label_general'] = labels_general
df.to_excel('sentences.xlsx', index=False)


# Make an English version
translator = Translator()
sentences_en = []
for sent in tqdm(sentences):
    sent_en = translator.translate(sent, src='ar', dest='en').text
    sentences_en.append(sent_en)
    time.sleep(1)

df = pd.DataFrame()
df['Sentence'] = sentences_en
df['Label'] = labels
df['Label_general'] = labels_general
df.to_csv('sentences.csv', index=False)

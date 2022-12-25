'''
Get Confusion matrices/fractions between annotations
'''

import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_annotations(file):
    annotations = []
    with open('{}'.format(file), encoding='utf-8') as json_file:
        data = json.load(json_file)
        for ann in data:
            annotations.append(ann['label'][0])
    return annotations


labels_speech = ["Speech", "Not Speech"] # Van Dijk - Discourse Profiling

labels_contents = ["Distant_Evaluation", "Distant_Expectations_Consequences", "Distant_Historical", "Distant_Anecdotal",
                   "Main", "Main_Consequence", "Cause_Specific", "Cause_General", "Other"] # Van Dijk - Discourse Profiling

labels_propaganda = ["Whataboutism,Straw_Men,Red_Herring", "Bandwagon,Reductio_ad_hitlerum", "Doubt", "Causal_Oversimplification",
                   "Black-and-White_Fallacy", "Appeal_to_Authority", "Appeal_to_fear-prejudice",
                   "Exaggeration,Minimisation", "Flag-Waving", "Repetition", "Loaded_Language",
                   "Name_Calling,Labeling", "Thought-terminating_Cliches", "other-nonpropaganda",
                   "Slogans"] # Propaganda

labels_argumenation = ["assumption", "common-ground", "testimony", "anecdote", "statistics", "other"] # Argumentation Dataset


annotations_speech = get_annotations(file='van_dijk_speech.json')
annotations_content = get_annotations(file='van_dijk_content.json')
annotations_propaganda = get_annotations(file='propaganda.json')
annotations_argumentation = get_annotations(file='argumentation.json')


# Confusion matrix - Van Dijk Contents vs Van Dijk Speeches
cnf = confusion_matrix(annotations_content, annotations_speech, labels=labels_contents + labels_speech)
cnf_sub = cnf[:9, -2:].T
cmn = cnf_sub.astype('float') / cnf_sub.sum(axis=1)[:, np.newaxis]
ax = sns.heatmap(cmn, annot=True, fmt = '.2f', xticklabels=labels_contents, yticklabels=labels_speech, square=True)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=45, fontsize=12)
# plt.show()
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(12, 6)
plt.savefig('contents_speeches.png', dpi=300)
plt.savefig('contents_speeches.pdf', dpi=300)
plt.close()

# Confusion Matrix - Van Dijk contents vs Argumentation
cnf = confusion_matrix(annotations_content, annotations_argumentation, labels=labels_contents + labels_argumenation)
print(cnf.sum())
cnf_sub = cnf[:9, 9:].T
print(cnf_sub.sum())
cmn = cnf_sub.astype('float') / cnf_sub.sum(axis=1)[:, np.newaxis]
ax = sns.heatmap(cmn, annot=True, fmt = '.2f', xticklabels=labels_contents, yticklabels=labels_argumenation, square=True)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=45, fontsize=12)
# plt.show()
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig('contents_argumentation.png', dpi=300)
plt.savefig('contents_argumentation.pdf', dpi=300)
plt.close()

# Confusion Matrix - Van Dijk contents vs Propaganda
cnf = confusion_matrix(annotations_content, annotations_propaganda, labels=labels_contents + labels_propaganda)
print(cnf.sum())
cnf_sub = cnf[:9, 9:]
print(cnf_sub.sum())
cmn = cnf_sub.astype('float') / cnf_sub.sum(axis=1)[:, np.newaxis]
ax = sns.heatmap(cmn, annot=True, fmt = '.2f', xticklabels=labels_propaganda, yticklabels=labels_contents, square=True)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
plt.xticks(rotation=90, fontsize=14)
plt.yticks(rotation=45, fontsize=14)
fig = plt.gcf()
fig.set_size_inches(20, 14)
plt.savefig('contents_propaganda.png', dpi=300)
plt.savefig('contents_propaganda.pdf', dpi=300)
plt.close()
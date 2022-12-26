import json
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

# associate each sentence with a year
# get labels per sentence
# grouped bar plot of labels across time
# check the old paper of sense disambiguation they also made sth similar but using clustering
# check the DiscoScore paper


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_annotations(file, sentences):
    annotations = []
    with open('{}'.format(file), encoding='utf-8') as json_file:
        data = json.load(json_file)
        for i, ann in enumerate(data):
            id = ann["id"]
            text = ann["text"]
            label = ann["label"][0]
            annotations.append({
                "id": id,
                "text": text,
                "label": label,
                "year": sentences[i][1],
                "group": sentences[i][2]
            })
    return annotations


group = ''
year = 0
sentences_meta_speech = {}
sentences_meta_content = {}
sentences_meta_propaganda = {}
sentences_meta_argumentation = {}
sentences = []
years = []
groups = []
group2name = {
    'group 0': 'Palestinian Resistance South Lebanon',
    'group 1': 'Sabra and Shatila Massacre'
}

name2savename = {
    'Palestinian Resistance South Lebanon': 'palestinian_resistance_south_lebanon',
    'Sabra and Shatila Massacre': 'sabra_shatila_massacre'
}

with open('sentences.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

    for line in lines:
        if line.strip() == '':
            continue
        if 'group' in line.strip():
            year, group = line.strip().split('-')
            year = int(year.strip())
            group = group.strip()
            group = group2name[group]
            continue

        if line.strip().isdigit():
            continue

        else:
            sentences.append((line.strip(), year, group))
            years.append(year)
            groups.append(group)

annotations_speech = get_annotations(file='van_dijk_speech.json', sentences=sentences)
annotations_content = get_annotations(file='van_dijk_content.json', sentences=sentences)
annotations_propaganda = get_annotations(file='propaganda.json', sentences=sentences)
annotations_argumentation = get_annotations(file='argumentation.json', sentences=sentences)

groups_unique = list(set(groups))
years_unique = sorted(list(set(years)))

shift_types = [annotations_content, annotations_argumentation, annotations_propaganda, annotations_speech]
shift_types_names = ["Van_Dijk_Contents", "Argumentation", "Propaganda", "Van_Dijk_Speeches"]

# create directory for saving plots
save_dir = 'shifts_plots/'
mkdir(save_dir)

for i, shift_type in enumerate(shift_types):
    annotations_shifts = {}
    # loop over every group
    for group in groups_unique:
        annotations_shifts[group] = {}
        # loop over every year:
        for year in years_unique:
            # get all annotations for a particular group-year
            annotations_shifts[group][year] = {}
            for ann in shift_type:
                if ann["group"] == group and ann["year"] == year:
                    label = ann["label"]
                    if label in annotations_shifts[group][year]:
                        annotations_shifts[group][year][label] += 1
                    else:
                        annotations_shifts[group][year][label] = 1

        # display grouped bar plots for each group
        d = annotations_shifts[group]
        to_plot = []
        for y in d:
            temp = d[y]
            # temp = dict(sorted(temp.items(), key=lambda kv: kv[1], reverse=True))

            tempV = np.array(list(temp.values()))
            tempV = tempV / tempV.sum()

            tempP = {}
            count = 0
            for k in temp:
                tempP[k] = tempV[count]
                count += 1
            tempP = dict(sorted(tempP.items(), key=lambda kv: kv[1], reverse=True))
            to_plot.append(tempP)

        df = pd.DataFrame(to_plot)

        df.plot(kind="bar", stacked=True)
        fig = plt.gcf()
        fig.set_size_inches(20, 14)
        ax = plt.subplot(111)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=6, fancybox=True, shadow=True)
        ax.set_xticklabels(years_unique)
        # ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks(rotation=45)
        plt.ylabel('Discourse Shifts in {}'.format(group))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(shift_types_names[i], name2savename[group])), dpi=300)
        plt.savefig(os.path.join(save_dir, '{}_{}.pdf'.format(shift_types_names[i], name2savename[group])), dpi=300)
        plt.close()
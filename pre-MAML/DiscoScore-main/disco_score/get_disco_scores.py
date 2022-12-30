import os
import nltk
from nltk import *
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
import json
from scorer import DiscoScorer
import pandas as pd
import numpy as np
from sklearn import preprocessing


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


def get_distribution(annotations):
    # get unique years
    # get unique labels
    years, labels = set(), set()
    for ann in annotations:
        years.add(ann["year"])
        labels.add(ann["label"])

    years = sorted(list(years))
    labels = list(labels)

    labels2i = {label: i for i, label in enumerate(labels)}
    years2i = {year: i for i, year in enumerate(years)}

    distributions = np.zeros((len(labels), len(years)))

    for ann in annotations:
        year = ann["year"]
        label = ann["label"]

        distributions[labels2i[label], years2i[year]] += 1

    return distributions

def entropy_timeseries(usage_distribution, intervals=None):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :return: array of entropy values, one for each usage distribution
    """
    if intervals:
        usage_distribution = usage_distribution[:, intervals]

    usage_distribution = preprocessing.normalize(usage_distribution, norm='l1', axis=0)

    H = []
    for t in range(usage_distribution.shape[1]):
        c = usage_distribution[:, t]
        if any(c):
            h = entropy(c)
        else:
            continue  # h = 0.
        H.append(h)

    return np.array(H)


def entropy_difference_timeseries(usage_distribution, absolute=True, intervals=None):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :return: array of entropy differences between contiguous usage distributions
    """
    if absolute:
        return np.array([abs(d) for d in np.diff(entropy_timeseries(usage_distribution, intervals))])
    else:
        return np.diff(entropy_timeseries(usage_distribution, intervals))


def js_divergence(*usage_distribution):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :return: Jensen-Shannon Divergence between multiple usage distributions
    """
    clusters = np.vstack(usage_distribution)
    n = clusters.shape[1]
    entropy_of_sum = entropy(1 / n * np.sum(clusters, axis=1))
    sum_of_entropies = 1 / n * np.sum([entropy(clusters[:, t]) for t in range(n)])
    return entropy_of_sum - sum_of_entropies


def js_distance(*usage_distribution):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :return: Jensen-Shannon Distance between two usage distributions
    """
    return np.sqrt(js_divergence(usage_distribution))


def jsd_timeseries(usage_distribution, dfunction=js_divergence, intervals=None):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :param dfunction: a JSD function (js_divergence or js_distance)
    :return: array of JSD between contiguous usage distributions
    """
    if intervals:
        usage_distribution = usage_distribution[:, intervals]

    usage_distribution = preprocessing.normalize(usage_distribution, norm='l1', axis=0)
    distances = []
    for t in range(usage_distribution.shape[1] - 1):
        c = usage_distribution[:, t]
        c_next = usage_distribution[:, t + 1]

        if any(c) and any(c_next):
            d = dfunction(c_next, c)
        else:
            continue  # d = 0.
        distances.append(d)

    return np.array(distances)


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

shift2ann = {
    "Van_Dijk_Contents": annotations_content,
    "Argumentation": annotations_argumentation,
    "Propaganda": annotations_propaganda,
    "Van_Dijk_Speeches": annotations_speech
}

# create directory for saving plots
save_dir = 'shifts_plots/'
mkdir(save_dir)

annotations_shifts = {}
for i, shift_type in enumerate(shift_types):
    shift_type_name = shift_types_names[i]
    annotations_shifts[shift_type_name] = {}
    # loop over every group
    for group in groups_unique:
        annotations_shifts[shift_type_name][group] = {}
        # loop over every year:
        for year in years_unique:
            # get all annotations for a particular group-year
            annotations_shifts[shift_type_name][group][year] = []
            for ann in shift_type:
                if ann["group"] == group and ann["year"] == year:
                    text = ann["text"]
                    annotations_shifts[shift_type_name][group][year].append(text)


# consider JSD and Entropy as the ground truth - per year
# plot coherence general vs coherenec ereference vs JSD/Entropy to see correlation between coherence and change in probability distributions
# add paired t-test to check difference between general and specific coherence
# bias? detect bias (1) inside a single discourse group (2) vs. across all the discourse groups as a whole
# do above for every year

# usage_distr = get_distribution(annotations_propaganda)
# usage_distr = preprocessing.normalize(usage_distr, norm='l1', axis=0)
#
# intervals = [5, 6, 7, 8]
# interval_labels = [1960, 1970, 1980, 1990]
#         # intervals = [5, 8]
#         # interval_labels = [1960, 1990]
#
#         # JSD
# jsd = jsd_timeseries(usage_distr, dfunction=js_divergence) / usage_distr.shape[0]
# jsd_multi.append(js_divergence([usage_distr[:, t] for t in intervals]))
# jsd_mean.append(np.mean(jsd))
# jsd_max.append(np.max(jsd))
# jsd_min.append(np.min(jsd))
# jsd_median.append(np.median(jsd))
#
# # Entropy difference
# dh = entropy_difference_timeseries(usage_distr, absolute=False, intervals=intervals) / usage_distr.shape[0]
# dh_mean.append(np.mean(dh))
# dh_max.append(np.max(dh))
# dh_min.append(np.min(dh))
# dh_median.append(np.median(dh))



# this hypothesis is 1982 based
# hypothesis = 'لذلك فان تهديدات اسرائيل بضرب المقاومه الفلسطينيه بحجه انها تخرق وقف النار تسمح بخرقه الجنوب مكان العالم تستهدف المقاومه بقدر تستهدف الجنوب نفسه تحقيقا لاهدافها وهي تهديدات ستنفذها الوقت المناسب حاجاتها الخاصه'
hypothesis = 'واكد ان اسرائيل هي المسؤول الاول والاخير مجزره مخيمي صبرا وشاتيلا'


# disco_scorer = DiscoScorer(device='cpu', model_name='bert-base-multilingual-cased')
# disco_scorer = DiscoScorer(device='cpu', model_name='bert-base-uncased')
disco_scorer = DiscoScorer(device='cpu', model_name='aubmindlab/bert-base-arabertv2')
# group = 'Palestinian Resistance South Lebanon'
group = 'Sabra and Shatila Massacre'
# annotation_style = "Propaganda"
df = pd.DataFrame(columns=['annotation', '1982', '1984', '1985', '1986', '1987'])

for annotation_style in shift_types_names:

    print('annotations style: {}'.format(annotation_style))

    coherences_general, coherences_specific = [], []

    refs1 = annotations_shifts[annotation_style][group][1982]
    refs2 = annotations_shifts[annotation_style][group][1984]
    refs3 = annotations_shifts[annotation_style][group][1985]
    refs4 = annotations_shifts[annotation_style][group][1986]
    refs5 = annotations_shifts[annotation_style][group][1987]

    # print('DiscoScore 1982: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs1)))  # FocusDiff
    # print('DiscoScore 1984: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs2))) # FocusDiff
    # print('DiscoScore 1985: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs3))) # FocusDiff
    # print('DiscoScore 1986: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs4))) # FocusDiff
    # print('DiscoScore 1987: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs5))) # FocusDiff

    annotations = shift2ann[annotation_style]

    # get the score when having the same annotation
    for ann in annotations:
        if ann["text"] == hypothesis.strip():
            label = ann["label"]
            break

    refs1_sub = []
    for r in refs1:
        for ann in annotations:
            if ann["text"] == r.strip() and ann["label"] == label:
                refs1_sub.append(r)
                break
    print('{} out of {}'.format(len(refs1_sub), len(refs1)))

    cg = disco_scorer.DS_Focus_NN(hypothesis, refs1)
    cs = disco_scorer.DS_Focus_NN(hypothesis, refs1_sub)

    print('DiscoScore 1982: {}'.format(cg))  # FocusDiff
    print('DiscoScore 1982: {}'.format(cs))  # FocusDiff

    coherences_general.append(str(cg))
    coherences_specific.append(str(cs))

    if len(refs1_sub) <= 5:
        print('hypothesis: {}'.format(hypothesis))
        for i, s in enumerate(refs1_sub):
            print('s {}: {}'.format(i+1, s))

    refs2_sub = []
    for r in refs2:
        for ann in annotations:
            if ann["text"] == r.strip() and ann["label"] == label:
                refs2_sub.append(r)
                break
    print('{} out of {}'.format(len(refs2_sub), len(refs2)))

    cg = disco_scorer.DS_Focus_NN(hypothesis, refs2)
    cs = disco_scorer.DS_Focus_NN(hypothesis, refs2_sub)

    print('DiscoScore 1984: {}'.format(cg))  # FocusDiff
    print('DiscoScore 1984: {}'.format(cs)) # FocusDiff

    coherences_general.append(str(cg))
    coherences_specific.append(str(cs))

    if len(refs2_sub) <= 5:
        print('hypothesis: {}'.format(hypothesis))
        for i, s in enumerate(refs2_sub):
            print('s {}: {}'.format(i+1, s))

    refs3_sub = []
    for r in refs3:
        for ann in annotations:
            if ann["text"] == r.strip() and ann["label"] == label:
                refs3_sub.append(r)
                break
    print('{} out of {}'.format(len(refs3_sub), len(refs3)))
    cg = disco_scorer.DS_Focus_NN(hypothesis, refs3)
    cs = disco_scorer.DS_Focus_NN(hypothesis, refs3_sub)

    print('DiscoScore 1985: {}'.format(cg))  # FocusDiff
    print('DiscoScore 1985: {}'.format(cs)) # FocusDiff

    coherences_general.append(str(cg))
    coherences_specific.append(str(cs))

    if len(refs3_sub) <= 5:
        print('hypothesis: {}'.format(hypothesis))
        for i, s in enumerate(refs3_sub):
            print('s {}: {}'.format(i+1, s))

    refs4_sub = []
    for r in refs4:
        for ann in annotations:
            if ann["text"] == r.strip() and ann["label"] == label:
                refs4_sub.append(r)
                break
    print('{} out of {}'.format(len(refs4_sub), len(refs4)))
    cg = disco_scorer.DS_Focus_NN(hypothesis, refs4)
    cs = disco_scorer.DS_Focus_NN(hypothesis, refs4_sub)

    print('DiscoScore 1986: {}'.format(cg))  # FocusDiff
    print('DiscoScore 1986: {}'.format(cs)) # FocusDiff

    coherences_general.append(str(cg))
    coherences_specific.append(str(cs))

    if len(refs4_sub) <= 5:
        print('hypothesis: {}'.format(hypothesis))
        for i, s in enumerate(refs4_sub):
            print('s {}: {}'.format(i+1, s))


    refs5_sub = []
    for r in refs5:
        for ann in annotations:
            if ann["text"] == r.strip() and ann["label"] == label:
                refs5_sub.append(r)
                break
    print('{} out of {}'.format(len(refs5_sub), len(refs5)))
    cg = disco_scorer.DS_Focus_NN(hypothesis, refs5)
    cs = disco_scorer.DS_Focus_NN(hypothesis, refs5_sub)
    print('DiscoScore 1987: {}'.format(cg))  # FocusDiff
    print('DiscoScore 1987: {}'.format(cs)) # FocusDiff

    coherences_general.append(str(cg))
    coherences_specific.append(str(cs))

    if len(refs5_sub) <= 5:
        print('hypothesis: {}'.format(hypothesis))
        for i, s in enumerate(refs5_sub):
            print('s {}: {}'.format(i+1, s))

    df = df.append({
        'annotation': '{}_general'.format(annotation_style),
        '1982': coherences_general[0],
        '1984': coherences_general[1],
        '1985': coherences_general[2],
        '1986': coherences_general[3],
        '1987': coherences_general[4]
    }, ignore_index=True)

    df = df.append({
        'annotation': '{}_specific'.format(annotation_style),
        '1982': coherences_specific[0],
        '1984': coherences_specific[1],
        '1985': coherences_specific[2],
        '1986': coherences_specific[3],
        '1987': coherences_specific[4]
    }, ignore_index=True)

    df = df.append({
        'annotation': '',
        '1982': '',
        '1984': '',
        '1985': '',
        '1986': '',
        '1987': ''
    }, ignore_index=True)

    df.to_csv('results_{}.csv'.format(name2savename[group]), index=False)

    print('===================================================================================================================')

df.to_csv('results_{}.csv'.format(name2savename[group]), index=False)

# system = ["Paul Merson has restarted his row with andros townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley. Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley ."]
# references = [["Paul Merson has restarted his row with burnley on sunday. Townsend was brought on in the 83rd minute for tottenham. Andros Townsend scores england 's equaliser in their 1-1 friendly draw. Townsend hit a stunning equaliser for england against italy."]]
#
# for s, refs in zip(system, references):
#    s = s.lower()
#    # refs = [r.lower() for r in refs]
#    # print(disco_scorer.EntityGraph(s, refs))
#    # print(disco_scorer.LexicalChain(s, refs))
#    # print(disco_scorer.RC(s, refs))
#    # print(disco_scorer.LC(s, refs))
#    print(disco_scorer.DS_Focus_NN(s, refs)) # FocusDiff
#    # print(disco_scorer.DS_SENT_NN(s, refs)) # SentGraph



# C:\Users\96171\AppData\Local\Programs\Python\Python36\python.exe C:/Users/96171/Desktop/political_discourse_mining_hiyam/pre-MAML/DiscoScore-main/disco_score/get_disco_scores.py
# C:\Users\96171\AppData\Local\Programs\Python\Python36\lib\site-packages\requests\__init__.py:104: RequestsDependencyWarning: urllib3 (1.25.3) or chardet (5.0.0)/charset_normalizer (2.0.12) doesn't match a supported version!
#   RequestsDependencyWarning)
# Some weights of the model checkpoint at aubmindlab/bert-base-arabertv2 were not used when initializing BertModel: ['cls.predictions.decoder.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight']
# - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# 1 out of 15
# DiscoScore 1982: 0.07640937661003895
# DiscoScore 1982: 0.0
# 0 out of 8
# DiscoScore 1984: 0.05147233587400243
# DiscoScore 1984: nan
# 6 out of 79
# C:\Users\96171\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\fromnumeric.py:3373: RuntimeWarning: Mean of empty slice.
#   out=out, **kwargs)
# C:\Users\96171\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
#   ret = ret.dtype.type(ret / rcount)
# DiscoScore 1985: 0.08328885960570327
# DiscoScore 1985: 0.07521219669468293
# 6 out of 115
# DiscoScore 1986: 0.07822415574117911
# DiscoScore 1986: 0.12207263802143871
# 2 out of 90
# DiscoScore 1987: 0.07469119311757882
# DiscoScore 1987: 0.14872830253982758
#
# Process finished with exit code 0
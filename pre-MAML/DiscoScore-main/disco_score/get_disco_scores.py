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
import scipy.stats as stats
from scipy.stats import entropy


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

def plot_disco_scores(df, group_name, save_dir):
    annotation_styles, annotation_vals = [], []
    cols = [col for col in df.columns if col != 'annotation']
    for i, row in df.iterrows():
        if 'specific' in str(row['annotation']):
            annotation_style = row['annotation']
            vals = []
            for col in cols:
                if '*' in str(row[col]):
                    vals.append(np.float(str(row[col]).replace('*', '')))
                else:
                    vals.append(np.float(row[col]))
            vals = [np.float(row[col]) for col in cols]
            annotation_styles.append(annotation_style)
            annotation_vals.append(vals)
    import itertools
    import matplotlib.pyplot as plt
    marker = itertools.cycle(('p', 's', 'D', 'X'))

    for i, style in enumerate(annotation_styles):
        plt.plot(list(range(len(annotation_vals[i]))), annotation_vals[i], marker=next(marker), label=annotation_styles[i])

    plt.xticks(list(range(len(annotation_vals[0]))), cols)
    plt.ylim(-0.5, 1.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True, prop={'size': 6})
    plt.xlabel('Year')
    plt.ylabel('DiscoScore')
    plt.tight_layout()
    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, group_name + '.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, group_name + '.pdf'), dpi=300)
    plt.close()


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
    'Sabra and Shatila Massacre': 'sabra_shatila_massacre',
    'Palestinian Resistance South Lebanon': 'palestinian_resistance_south_lebanon',
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
# save_dir = 'shifts_plots/'
# mkdir(save_dir)

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


# DONE: consider JSD and Entropy as the ground truth - per year
# DONE: plot coherence general vs coherenec ereference vs JSD/Entropy to see correlation between coherence and change in probability distributions
# DONE: add paired t-test to check difference between general and specific coherence
# results between specific and general are not statistically significant
# however there is a change in the coherence in reference to an initial hypothesis - by taking all
# the sentences that have the same label as the hypothesis, across all years i.e. discourse is changing


# bias? detect bias (1) inside a single discourse group (2) vs. across all the discourse groups as a whole
# do above for every year
# the choice of the hypothesis is to choose an opinionated one (discousre having an ideology - works with
# Distant_evaluation/argumentation/propaganda, etc) and see if coherence in an ideology is maintained

usage_distr = get_distribution(annotations_content)
# usage_distr = preprocessing.normalize(usage_distr, norm='l1', axis=0)
#
# intervals = [5, 6, 7, 8]
# interval_labels = [1960, 1970, 1980, 1990]
#         # intervals = [5, 8]
#         # interval_labels = [1960, 1990]
#
#         # JSD
# jsd = jsd_timeseries(usage_distr, dfunction=js_divergence) / usage_distr.shape[0]
# print('jsd: {}'.format(jsd))
# jsd_multi.append(js_divergence([usage_distr[:, t] for t in intervals]))
# jsd_mean.append(np.mean(jsd))
# jsd_max.append(np.max(jsd))
# jsd_min.append(np.min(jsd))
# jsd_median.append(np.median(jsd))
#
# # Entropy difference
print('content')
usage_distr = get_distribution(annotations_content)
dh = entropy_timeseries(usage_distr) / usage_distr.shape[0]
print('ed: {}'.format(dh))
print('ed mean: {}'.format(np.mean(dh)))
print('ed max: {}'.format(np.max(dh)))
print('ed min: {}'.format(np.min(dh)))
print('ed median: {}'.format(np.median(dh)))

print('argumentation')
usage_distr = get_distribution(annotations_argumentation)
dh = entropy_timeseries(usage_distr) / usage_distr.shape[0]
print('ed: {}'.format(dh))
print('ed mean: {}'.format(np.mean(dh)))
print('ed max: {}'.format(np.max(dh)))
print('ed min: {}'.format(np.min(dh)))
print('ed median: {}'.format(np.median(dh)))

print('propaganda')
usage_distr = get_distribution(annotations_propaganda)
dh = entropy_timeseries(usage_distr) / usage_distr.shape[0]
print('ed: {}'.format(dh))
print('ed mean: {}'.format(np.mean(dh)))
print('ed max: {}'.format(np.max(dh)))
print('ed min: {}'.format(np.min(dh)))
print('ed median: {}'.format(np.median(dh)))

print('speech')
usage_distr = get_distribution(annotations_speech)
dh = entropy_timeseries(usage_distr) / usage_distr.shape[0]
print('ed: {}'.format(dh))
print('ed mean: {}'.format(np.mean(dh)))
print('ed max: {}'.format(np.max(dh)))
print('ed min: {}'.format(np.min(dh)))
print('ed median: {}'.format(np.median(dh)))



# this hypothesis is 1982 based
# hypothesis = 'لذلك فان تهديدات اسرائيل بضرب المقاومه الفلسطينيه بحجه انها تخرق وقف النار تسمح بخرقه الجنوب مكان العالم تستهدف المقاومه بقدر تستهدف الجنوب نفسه تحقيقا لاهدافها وهي تهديدات ستنفذها الوقت المناسب حاجاتها الخاصه'
hypothesis = 'واكد ان اسرائيل هي المسؤول الاول والاخير مجزره مخيمي صبرا وشاتيلا'

group2hypothesis = {
    'Sabra and Shatila Massacre': 'واكد ان اسرائيل هي المسؤول الاول والاخير مجزره مخيمي صبرا وشاتيلا',
    'Palestinian Resistance South Lebanon': 'لذلك فان تهديدات اسرائيل بضرب المقاومه الفلسطينيه بحجه انها تخرق وقف النار تسمح بخرقه الجنوب مكان العالم تستهدف المقاومه بقدر تستهدف الجنوب نفسه تحقيقا لاهدافها وهي تهديدات ستنفذها الوقت المناسب حاجاتها الخاصه'
}

# name2savename = {
#     'Palestinian Resistance South Lebanon': 'palestinian_resistance_south_lebanon',
#     'Sabra and Shatila Massacre': 'sabra_shatila_massacre'
# }


# disco_scorer = DiscoScorer(device='cpu', model_name='bert-base-multilingual-cased')
# disco_scorer = DiscoScorer(device='cpu', model_name='bert-base-uncased')
disco_scorer = DiscoScorer(device='cpu', model_name='aubmindlab/bert-base-arabertv2')
# group = 'Palestinian Resistance South Lebanon'
group = 'Sabra and Shatila Massacre'
# annotation_style = "Propaganda"


years = list(sorted(list(annotations_shifts[annotation_style][group].keys())))
ref2score = {}
for group in group2hypothesis:
    years = [1982, 1984, 1985, 1986, 1987]
    hypothesis = group2hypothesis[group]
    ref2score[group] = {}
    for year in years:
        scores, idxs = [], []
        refs = annotations_shifts["Propaganda"][group][year]
        i = 0
        for ref in refs:
            score = disco_scorer.DS_Focus_NN(hypothesis, [ref])
            scores.append(score[0])
            ref2score[group][ref] = score[0]
            idxs.append(i)
            i += 1
        from operator import itemgetter

        res = [list(x) for x in zip(*sorted(zip(scores, idxs), key=itemgetter(0)))]
        res[0].reverse()
        res[1].reverse()

        scores = res[0]
        idxs = res[1]
        print(scores)
#
import pickle

# with open('ref2score.pickle', 'wb') as handle:
#     pickle.dump(ref2score, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('ref2score.pickle', 'rb') as handle:
    ref2score = pickle.load(handle)

# pick top 10 most coherent sentences in each year

for group in name2savename:

    print('group name: {}'.format(group))
    df = pd.DataFrame(columns=['annotation', '1982', '1984', '1985', '1986', '1987'])
    hypothesis = group2hypothesis[group]

    highest_corr = 0
    highest_ann = ''

    for annotation_style in shift2ann:

        print('annotations style: {}'.format(annotation_style))

        coherences_general, coherences_specific = [], []
        scores_general, scores_specific = [], []

        years = list(sorted(list(annotations_shifts[annotation_style][group].keys())))

        for year in years:

            refs = annotations_shifts[annotation_style][group][year]

            annotations = shift2ann[annotation_style]

            # get the score when having the same annotation
            for ann in annotations:
                if ann["text"] == hypothesis.strip():
                    label = ann["label"]
                    break

            refs_sub = []
            for r in refs:
                for ann in annotations:
                    if ann["text"] == r.strip() and ann["label"] == label:
                        refs_sub.append(r)
                        break
            print('{} out of {}'.format(len(refs_sub), len(refs)))

            list_gen = list(ref2score[group].values())
            list_spec = list(ref2score[group][r] for r in refs_sub)

            cg = np.mean(list_gen)
            cs = np.mean(list_spec)

            print('DiscoScore {}: {}'.format(year, cg))  # FocusDiff
            print('DiscoScore {}: {}'.format(year, cs))  # FocusDiff

            coherences_general.append(str(cg))
            if len(list_spec) > 0:
                coherences_specific.append(str(cs))
            else:
                coherences_specific.append(str(cg))

            scores_general.append(list_gen)
            scores_specific.append(list_spec)

            # res = stats.ttest_ind(scores_g, scores_s)
            # if res.pvalue < 0.05:
            #     print('difference in coherence scores between general and specific in year {} are statistically significant: {}'.format(year, res.pvalue))
            # else:
            #     print('difference in coherence NOT statistically significant FOR YEAR {}: {}'.format(year, res.pvalue))

            res2 = stats.mannwhitneyu(list_gen, list_spec)
            if res2.pvalue < 0.05:
                print('difference in coherence scores between general and specific in year {} are statistically significant: {}'.format(year, res2.pvalue))
            else:
                print('difference in coherence NOT statistically significant FOR YEAR {}: {}'.format(year, res2.pvalue))

            if len(refs_sub) <= 5:
                print('hypothesis: {}'.format(hypothesis))
                for i, s in enumerate(refs_sub):
                    print('s {}: {}'.format(i+1, s))

        data_gen = {'annotation': '{}_general'.format(annotation_style)}
        for i, year in enumerate(years):
            year = str(year)
            data_gen[year] = coherences_general[i]
        df = df.append(data_gen, ignore_index=True)

        data_sp = {'annotation': '{}_specific'.format(annotation_style)}
        for i, year in enumerate(years):
            year = str(year)
            # res = stats.ttest_ind(scores_general[i], scores_specific[i]) # check if differences in coherence scores are statistically significant
            res = stats.mannwhitneyu(scores_general[i], scores_specific[i])
            # if res.pvalue < 0.05:
            #     data_sp[year] = '*' + str(coherences_specific[i])
            # else:
            #     data_sp[year] = coherences_specific[i]
            data_sp[year] = coherences_specific[i]
        df = df.append(data_sp, ignore_index=True)

        from scipy.stats import pearsonr
        try:
            corr, _ = pearsonr(dh, [np.float(v) for v in coherences_specific])
            print('Pearsons correlation: %.3f' % corr)

            if corr > highest_corr:
                highest_corr = corr
                highest_ann = annotation_style
        except:
            print('CANNOTCALCULATE PEARSON')

        df = df.append({k: '' for k in data_sp}, ignore_index=True)

        df.to_csv('results1_{}.csv'.format(name2savename[group]), index=False)

        print('===================================================================================================================')

    df.to_csv('results1_{}.csv'.format(name2savename[group]), index=False)

    print('Highest corr: {} by {}'.format(highest_corr, highest_ann))

    plot_disco_scores(df=df, group_name=group, save_dir='coherence_plots/')


# for group in name2savename:
#
#     print('group name: {}'.format(group))
#     df = pd.DataFrame(columns=['annotation', '1982', '1984', '1985', '1986', '1987'])
#     hypothesis = group2hypothesis[group]
#
#     for annotation_style in shift2ann:
#
#         print('annotations style: {}'.format(annotation_style))
#
#         coherences_general, coherences_specific = [], []
#         scores_general, scores_specific = [], []
#
#         years = list(sorted(list(annotations_shifts[annotation_style][group].keys())))
#
#         for year in years:
#
#             refs = annotations_shifts[annotation_style][group][year]
#
#             annotations = shift2ann[annotation_style]
#
#             # get the score when having the same annotation
#             for ann in annotations:
#                 if ann["text"] == hypothesis.strip():
#                     label = ann["label"]
#                     break
#
#             refs_sub = []
#             for r in refs:
#                 for ann in annotations:
#                     if ann["text"] == r.strip() and ann["label"] == label:
#                         refs_sub.append(r)
#                         break
#             print('{} out of {}'.format(len(refs_sub), len(refs)))
#
#             cg, scores_g = disco_scorer.DS_Focus_NN(hypothesis, refs)
#             cs, scores_s = disco_scorer.DS_Focus_NN(hypothesis, refs_sub)
#
#             print('DiscoScore {}: {}'.format(year, cg))  # FocusDiff
#             print('DiscoScore {}: {}'.format(year, cs))  # FocusDiff
#
#             coherences_general.append(str(cg))
#             coherences_specific.append(str(cs))
#
#             scores_general.append(scores_g)
#             scores_specific.append(scores_s)
#
#             # res = stats.ttest_ind(scores_g, scores_s)
#             # if res.pvalue < 0.05:
#             #     print('difference in coherence scores between general and specific in year {} are statistically significant: {}'.format(year, res.pvalue))
#             # else:
#             #     print('difference in coherence NOT statistically significant FOR YEAR {}: {}'.format(year, res.pvalue))
#
#             res2 = stats.mannwhitneyu(scores_g, scores_s)
#             if res2.pvalue < 0.05:
#                 print('difference in coherence scores between general and specific in year {} are statistically significant: {}'.format(year, res2.pvalue))
#             else:
#                 print('difference in coherence NOT statistically significant FOR YEAR {}: {}'.format(year, res2.pvalue))
#
#             if len(refs_sub) <= 5:
#                 print('hypothesis: {}'.format(hypothesis))
#                 for i, s in enumerate(refs_sub):
#                     print('s {}: {}'.format(i+1, s))
#
#         data_gen = {'annotation': '{}_general'.format(annotation_style)}
#         for i, year in enumerate(years):
#             year = str(year)
#             data_gen[year] = coherences_general[i]
#         df = df.append(data_gen, ignore_index=True)
#
#         data_sp = {'annotation': '{}_specific'.format(annotation_style)}
#         for i, year in enumerate(years):
#             year = str(year)
#             # res = stats.ttest_ind(scores_general[i], scores_specific[i]) # check if differences in coherence scores are statistically significant
#             res = stats.mannwhitneyu(scores_general[i], scores_specific[i])
#             if res.pvalue < 0.05:
#                 data_sp[year] = '*' + str(coherences_specific[i])
#             else:
#                 data_sp[year] = coherences_specific[i]
#         df = df.append(data_sp, ignore_index=True)
#
#         df = df.append({k: '' for k in data_sp}, ignore_index=True)
#
#         df.to_csv('results1_{}.csv'.format(name2savename[group]), index=False)
#
#         print('===================================================================================================================')
#
#     df.to_csv('results1_{}.csv'.format(name2savename[group]), index=False)
#
#     plot_disco_scores(df=df, group_name=group, save_dir='coherence_plots/')


# # system = ["Paul Merson has restarted his row with andros townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley. Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley ."]
# # references = [["Paul Merson has restarted his row with burnley on sunday. Townsend was brought on in the 83rd minute for tottenham. Andros Townsend scores england 's equaliser in their 1-1 friendly draw. Townsend hit a stunning equaliser for england against italy."]]
# #
# # for s, refs in zip(system, references):
# #    s = s.lower()
# #    # refs = [r.lower() for r in refs]
# #    # print(disco_scorer.EntityGraph(s, refs))
# #    # print(disco_scorer.LexicalChain(s, refs))
# #    # print(disco_scorer.RC(s, refs))
# #    # print(disco_scorer.LC(s, refs))
# #    print(disco_scorer.DS_Focus_NN(s, refs)) # FocusDiff
# #    # print(disco_scorer.DS_SENT_NN(s, refs)) # SentGraph
#
#
#
# # C:\Users\96171\AppData\Local\Programs\Python\Python36\python.exe C:/Users/96171/Desktop/political_discourse_mining_hiyam/pre-MAML/DiscoScore-main/disco_score/get_disco_scores.py
# # C:\Users\96171\AppData\Local\Programs\Python\Python36\lib\site-packages\requests\__init__.py:104: RequestsDependencyWarning: urllib3 (1.25.3) or chardet (5.0.0)/charset_normalizer (2.0.12) doesn't match a supported version!
# #   RequestsDependencyWarning)
# # Some weights of the model checkpoint at aubmindlab/bert-base-arabertv2 were not used when initializing BertModel: ['cls.predictions.decoder.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight']
# # - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
# # - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
# # 1 out of 15
# # DiscoScore 1982: 0.07640937661003895
# # DiscoScore 1982: 0.0
# # 0 out of 8
# # DiscoScore 1984: 0.05147233587400243
# # DiscoScore 1984: nan
# # 6 out of 79
# # C:\Users\96171\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\fromnumeric.py:3373: RuntimeWarning: Mean of empty slice.
# #   out=out, **kwargs)
# # C:\Users\96171\AppData\Local\Programs\Python\Python36\lib\site-packages\numpy\core\_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
# #   ret = ret.dtype.type(ret / rcount)
# # DiscoScore 1985: 0.08328885960570327
# # DiscoScore 1985: 0.07521219669468293
# # 6 out of 115
# # DiscoScore 1986: 0.07822415574117911
# # DiscoScore 1986: 0.12207263802143871
# # 2 out of 90
# # DiscoScore 1987: 0.07469119311757882
# # DiscoScore 1987: 0.14872830253982758
# #
# # Process finished with exit code 0
import os
import nltk
from nltk import *
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
import json
from scorer import DiscoScorer


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


java_path = 'C:/Program Files/Java/jdk1.8.0_211/bin/'
os.environ["JAVAHOME"] = java_path

# this hypothesis is 1982 based
hypothesis = 'لذلك فان تهديدات اسرائيل بضرب المقاومه الفلسطينيه بحجه انها تخرق وقف النار تسمح بخرقه الجنوب مكان العالم تستهدف المقاومه بقدر تستهدف الجنوب نفسه تحقيقا لاهدافها وهي تهديدات ستنفذها الوقت المناسب حاجاتها الخاصه'

# disco_scorer = DiscoScorer(device='cpu', model_name='bert-base-multilingual-cased')
# disco_scorer = DiscoScorer(device='cpu', model_name='bert-base-uncased')
disco_scorer = DiscoScorer(device='cpu', model_name='aubmindlab/bert-base-arabertv2')
group = 'Palestinian Resistance South Lebanon'

# refs1 = annotations_shifts["Van_Dijk_Contents"][group][1982]
# refs2 = annotations_shifts["Van_Dijk_Contents"][group][1984]
# refs3 = annotations_shifts["Van_Dijk_Contents"][group][1985]
# refs4 = annotations_shifts["Van_Dijk_Contents"][group][1986]
# refs5 = annotations_shifts["Van_Dijk_Contents"][group][1987]

refs1 = annotations_shifts["Propaganda"][group][1982]
refs2 = annotations_shifts["Propaganda"][group][1984]
refs3 = annotations_shifts["Propaganda"][group][1985]
refs4 = annotations_shifts["Propaganda"][group][1986]
refs5 = annotations_shifts["Propaganda"][group][1987]

# refs1 = annotations_shifts["Argumentation"][group][1982]
# refs2 = annotations_shifts["Argumentation"][group][1984]
# refs3 = annotations_shifts["Argumentation"][group][1985]
# refs4 = annotations_shifts["Argumentation"][group][1986]
# refs5 = annotations_shifts["Argumentation"][group][1987]

# print('DiscoScore 1982: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs1)))  # FocusDiff
# print('DiscoScore 1984: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs2))) # FocusDiff
# print('DiscoScore 1985: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs3))) # FocusDiff
# print('DiscoScore 1986: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs4))) # FocusDiff
# print('DiscoScore 1987: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs5))) # FocusDiff

annotations = annotations_propaganda

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
print('DiscoScore 1982: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs1)))  # FocusDiff
print('DiscoScore 1982: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs1_sub)))  # FocusDiff

refs2_sub = []
for r in refs2:
    for ann in annotations:
        if ann["text"] == r.strip() and ann["label"] == label:
            refs2_sub.append(r)
            break
print('{} out of {}'.format(len(refs2_sub), len(refs2)))
print('DiscoScore 1984: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs2)))  # FocusDiff
print('DiscoScore 1984: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs2_sub))) # FocusDiff


refs3_sub = []
for r in refs3:
    for ann in annotations:
        if ann["text"] == r.strip() and ann["label"] == label:
            refs3_sub.append(r)
            break
print('{} out of {}'.format(len(refs3_sub), len(refs3)))
print('DiscoScore 1985: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs3)))  # FocusDiff
print('DiscoScore 1985: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs3_sub))) # FocusDiff

refs4_sub = []
for r in refs4:
    for ann in annotations:
        if ann["text"] == r.strip() and ann["label"] == label:
            refs4_sub.append(r)
            break
print('{} out of {}'.format(len(refs4_sub), len(refs4)))
print('DiscoScore 1986: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs4)))  # FocusDiff
print('DiscoScore 1986: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs4_sub))) # FocusDiff

refs5_sub = []
for r in refs5:
    for ann in annotations:
        if ann["text"] == r.strip() and ann["label"] == label:
            refs5_sub.append(r)
            break
print('{} out of {}'.format(len(refs5_sub), len(refs5)))
print('DiscoScore 1987: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs5)))  # FocusDiff
print('DiscoScore 1987: {}'.format(disco_scorer.DS_Focus_NN(hypothesis, refs5_sub))) # FocusDiff

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
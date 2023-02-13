import pandas as pd
from sklearn.metrics import *
import pickle
import os, json


def add_years_distribution():
    print()

def get_results(labels, preds, binary=False):
    if binary:
        accuracy = accuracy_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds)
        precision = precision_score(y_true=labels, y_pred=preds)
        recall = recall_score(y_true=labels, y_pred=preds)
        return {
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc_and_f1": (accuracy + f1) / 2,
        }

    else:
        accuracy = accuracy_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
        precision = precision_score(y_true=labels, y_pred=preds, average="macro")
        recall = recall_score(y_true=labels, y_pred=preds, average="macro")

        f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")
        precision_micro = precision_score(y_true=labels, y_pred=preds, average="micro")
        recall_micro = recall_score(y_true=labels, y_pred=preds, average="micro")

        f1_wei = f1_score(y_true=labels, y_pred=preds, average="weighted")
        precision_wei = precision_score(y_true=labels, y_pred=preds, average="weighted")
        recall_wei = recall_score(y_true=labels, y_pred=preds, average="weighted")

        return {
            "acc": accuracy,

            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,

            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,

            "precision_wei": precision_wei,
            "recall_wei": recall_wei,
            "f1_wei": f1_wei,
        }


def collate_results(rootdir, df_name, labels_list, df_test_years):
    df_results = pd.DataFrame(columns=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_micro', 'recall_micro',
                 'f1_micro', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'model', 'report'])

    lab2idx = {l: i for i, l in enumerate(labels_list)}

    for subdir, dirs, files in os.walk(rootdir):
        if 'results.pickle' in files:
            with open(os.path.join(subdir, 'results.pickle'), 'rb') as handle:
                results = pickle.load(handle)

                df_actvspred = pd.read_csv(os.path.join(subdir, 'test_actual_predicted.csv'))
                actual = list(df_actvspred['actual'])
                predicted = list(df_actvspred['predicted'])

                actual_idx = [lab2idx[l] for l in actual]
                predicted_idx = [lab2idx[l] for l in predicted]

                results_dict = get_results(labels=actual_idx, preds=predicted_idx)

                report = classification_report(actual_idx, predicted_idx, labels=list(lab2idx.values()), target_names=list(lab2idx.keys()), output_dict=True)
                report_str = json.dumps(report)

                df_results = df_results.append({
                    'accuracy': results['acc'],
                    'precision_macro': results['precision'],
                    'recall_macro': results['recall'],
                    'f1_macro': results['f1'],

                    'precision_micro': results_dict['precision_micro'],
                    'recall_micro': results_dict['recall_micro'],
                    'f1_micro': results_dict['f1_micro'],

                    'precision_weighted': results_dict['precision_wei'],
                    'recall_weighted': results_dict['recall_wei'],
                    'f1_weighted': results_dict['f1_wei'],

                    'model': subdir,
                    'report': report_str
                }, ignore_index=True)

    df_results = df_results.sort_values(by='accuracy', ascending=False)
    df_results.to_csv('{}.csv'.format(df_name), index=False)

    # get best results
    models = ['bert-base-multilingual-cased',
              'xlm-roberta-base',
              # 'xlm-roberta-large',
              'aubmindlab/bert-base-arabertv2',
              # 'aubmindlab/bert-large-arabertv02',
              'aubmindlab/bert-base-arabertv02',
              'distilbert-base-multilingual-cased']

    df_best_results = pd.DataFrame(columns=['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_micro', 'recall_micro',
                 'f1_micro', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'report'])
    model2yearsdist = {}
    for model in models:
        if 'aubmindlab' in model:
            model = '_'.join(model.split('/'))

        exps = []
        for i, row in df_results.iterrows():
            if model in row['model'] and exps == []:
                df_best_results = df_best_results.append({
                    'model': model + ' | zero shot' if 'ft0' in row['model'] else model + ' | few shot',

                    'accuracy': row['accuracy'],
                    'precision_macro': row['precision_macro'],
                    'recall_macro': row['recall_macro'],
                    'f1_macro': row['f1_macro'],

                    'precision_micro': row['precision_micro'],
                    'recall_micro': row['recall_micro'],
                    'f1_micro': row['f1_micro'],

                    'precision_weighted': row['precision_weighted'],
                    'recall_weighted': row['recall_weighted'],
                    'f1_weighted': row['f1_weighted'],

                    'report': row['report']
                }, ignore_index=True)

                if 'ft0' in row['model']:
                    exps.append('ft0')
                else:
                    exps.append('ft1')
            else:
                if model in row['model'] and row['model'].split('-')[-1].split('/')[0] not in exps:
                    df_best_results = df_best_results.append({
                        'model': model + ' | zero shot' if 'ft0' in row['model'] else model + ' | few shot',

                        'accuracy': row['accuracy'],
                        'precision_macro': row['precision_macro'],
                        'recall_macro': row['recall_macro'],
                        'f1_macro': row['f1_macro'],

                        'precision_micro': row['precision_micro'],
                        'recall_micro': row['recall_micro'],
                        'f1_micro': row['f1_micro'],

                        'precision_weighted': row['precision_weighted'],
                        'recall_weighted': row['recall_weighted'],
                        'f1_weighted': row['f1_weighted'],

                        'report': row['report']
                    }, ignore_index=True)

                    if 'ft0' in row['model']:
                        exps.append('ft0')
                    else:
                        exps.append('ft1')

            model_name = model + ' | zero shot' if 'ft0' in row['model'] else model + ' | few shot'
            subdir = row['model']
            df_actvspred = pd.read_csv(os.path.join(subdir, 'test_actual_predicted.csv'))
            df_actvspred['year'] = df_test_years['year']
            model2yearsdist[model_name] = df_actvspred

            if len(exps) == 2:
                break

    df_best_results.to_csv('{}_best.csv'.format(df_name), index=False)
    with open('{}_actvspred.pickle'.format(df_name), 'wb') as handle:
        pickle.dump(model2yearsdist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # get detailed results per class for every model
    df_detailed_results = pd.DataFrame(columns=['model'] + labels_list)
    for i, row in df_best_results.iterrows():
        results_sub = {k: '' for k in df_detailed_results.columns}
        results_sub['model'] = row['model']
        report = json.loads(row['report'])
        for label in labels_list:
            results_sub[label] = report[label]['f1-score']

        df_detailed_results = df_detailed_results.append(results_sub, ignore_index=True)

    df_detailed_results.to_csv('{}_detailed.csv'.format(df_name), index=False)


if __name__ == '__main__':

    df_test_years = pd.read_excel('../translate_corpora/annotations_doccano/translationsv2/propaganda/Sabra and Shatila Massacre/our_corpus_ar.xlsx')
    with open('groups2years_sentences.pickle', 'rb') as handle:
        group2years_sentences = pickle.load(handle)

    group = 'Sabra and Shatila Massacre'
    years = []
    for i, row in df_test_years.iterrows():
        sentence = row['Sentence'].strip()
        for year in group2years_sentences[group]:
            if sentence in group2years_sentences[group][year]:
                years.append(year)
                break

    df_test_years['year'] = years

    labels_ARG = "assumption,anecdote,testimony,statistics,common-ground,other"
    labels_ARG_corp = "assumption,statistics,other,testimony,common-ground,anecdote"
    # dropped no-unit in labels_ARG

    labels_VDC = "Main_Consequence,Cause_General,Cause_Specific,Distant_Expectations_Consequences,Distant_Historical,Main,Distant_Anecdotal,Distant_Evaluation"
    labels_VDC_corp = "Main_Consequence,Distant_Evaluation,Cause_Specific,Distant_Anecdotal,Distant_Expectations_Consequences,Main,Distant_Historical,Cause_General"
    # dropped nan in labels_VDC
    # dropped Other in labels_VDC_corp - as it indicated small parts of sentences that are there due to OCR

    labels_VDS = "Speech,Not Speech"
    labels_VDS_corp = "Speech,Not Speech"

    labels_PTC = "Causal_Oversimplification;Thought-terminating_Cliches;Appeal_to_fear-prejudice;Bandwagon,Reductio_ad_hitlerum;Exaggeration,Minimisation;Slogans;Black-and-White_Fallacy;Appeal_to_Authority;Name_Calling,Labeling;Flag-Waving;Doubt;Loaded_Language;Whataboutism,Straw_Men,Red_Herring;Repetition"
    labels_PTC_corp = "Black-and-White_Fallacy;Whataboutism,Straw_Men,Red_Herring;Flag-Waving;Causal_Oversimplification;Thought-terminating_Cliches;Exaggeration,Minimisation;Bandwagon,Reductio_ad_hitlerum;Name_Calling,Labeling;Appeal_to_fear-prejudice;Doubt;Repetition;Appeal_to_Authority;Loaded_Language;other-nonpropaganda"
    # will keep the other-nonpropaganda from labels_PTC_corp (its not found in labels_PTC though) because
    # we are interested in classifying non-propaganda instances, and the model will
    # be exposed to it during meta-training because every domain will be there

    # all labels are the union of the labels from X dataset and the labels from our corpus (our annotation scheme)
    all_labels_ARG = list(set(labels_ARG.split(",")).union(labels_ARG_corp.split(",")))
    all_labels_VDC = list(set(labels_VDC.split(",")).union(labels_VDC_corp.split(",")))
    all_labels_VDS = list(set(labels_VDS.split(",")).union(labels_VDS_corp.split(",")))
    all_labels_PTC = list(set(labels_PTC.split(";")).union(labels_PTC_corp.split(";")))

    collate_results(rootdir='results/ARG/', df_name='results_ARG', labels_list=all_labels_ARG, df_test_years=df_test_years)
    collate_results(rootdir='results/PTC/', df_name='results_PTC', labels_list=all_labels_PTC, df_test_years=df_test_years)
    collate_results(rootdir='results/VDC/', df_name='results_VDC', labels_list=all_labels_VDC, df_test_years=df_test_years)
    collate_results(rootdir='results_cross_lingaul/VDC/', df_name='results_VDC_cross_lingual', labels_list=all_labels_VDC, df_test_years=df_test_years)



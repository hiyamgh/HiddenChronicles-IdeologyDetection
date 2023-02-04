from utils_xmaml import codes2names
import os


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    save_dir = "cross_lingual_experiments/1_aux/"
    mkdir(save_dir)

    # define hyperparameters
    support_sizes_grid = [8, 16, 32]
    inner_train_steps_grid = [3, 5, 7, 10]
    inner_lr_grid = [1e-4, 1e-5, 3e-4, 3e-5, 5e-4, 5e-5]
    models = ['bert-base-multilingual-cased', 'xlm-roberta-base']

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

    # Discourse profiling - contexts
    # (meta train) corp_PRST_aux_VDC / domain 1
    # (fine tune) corp_PRST_ar_VDC / domain 1
    # (test) corp_SSM_ar_VDC

    with open(os.path.join(save_dir, 'VDC_cross_lingual.txt'), 'w') as f:
        for lang in codes2names:
            if lang != "ar":
                dev_dataset_ids = ['corp_PRST_{}_VDC'.format(lang)]
                dev_dataset_fine_tune_id = ['corp_PRST_ar_VDC']
                test_dataset_id = ['corp_SSM_ar_VDC']

                for n in support_sizes_grid:
                    for inner_train_step in inner_train_steps_grid:
                        for inner_lr in inner_lr_grid:
                            for model in models:
                                f.write(
                                    "--bert_model {} --dev_datasets_ids {} --dev_dataset_finetune {} --test_dataset_eval {} --n {} --inner_train_steps {} --inner_lr {} --labels {}\n".format(
                                        model,
                                        ",".join(dev_dataset_ids),
                                        ",".join(dev_dataset_fine_tune_id),
                                        ",".join(test_dataset_id),
                                        n, inner_train_step, inner_lr,
                                        ",".join(all_labels_VDC)
                                    ))
    f.close()


    # Discourse profiling - speeches
    # (meta train) corp_PRST_aux_VDS / domain 1
    # (fine tune) corp_PRST_ar_VDS / domain 1
    # (test) corp_SSM_ar_VDS

    with open(os.path.join(save_dir, 'VDS_cross_lingual.txt'), 'w') as f:
        for lang in codes2names:
            if lang != "ar":
                dev_dataset_ids = ['corp_PRST_{}_VDS'.format(lang)]
                dev_dataset_fine_tune_id = ['corp_PRST_ar_VDS']
                test_dataset_id = ['corp_SSM_ar_VDS']

                for n in support_sizes_grid:
                    for inner_train_step in inner_train_steps_grid:
                        for inner_lr in inner_lr_grid:
                            for model in models:
                                f.write(
                                    "--bert_model {} --dev_datasets_ids {} --dev_dataset_finetune {} --test_dataset_eval {} --n {} --inner_train_steps {} --inner_lr {} --labels {}\n".format(
                                        model,
                                        ",".join(dev_dataset_ids),
                                        ",".join(dev_dataset_fine_tune_id),
                                        ",".join(test_dataset_id),
                                        n, inner_train_step, inner_lr,
                                        ",".join(all_labels_VDS)
                                    ))
    f.close()

    # Argumentation
    # (meta train) corp_PRST_aux_ARG / domain 1
    # (fine tune) corp_PRST_ar_ARG / domain 1
    # (test) corp_SSM_ar_ARG

    # --n: number of support samples, query = batch size - n (16)
    # --inner_train_steps: number of inner updates (3)
    # --inner_lr: inner learning rate (alpha) (1e-4)
    # --learning_rate: outer learning rate (beta) (5e-5)
    # dev_datasets_ids
    # dev_dataset_finetune
    # test_dataset_eval

    with open(os.path.join(save_dir, 'argumentation_cross_lingual.txt'), 'w') as f:
        for lang in codes2names:
            if lang != "ar":
                dev_dataset_ids = ['corp_PRST_{}_ARG'.format(lang)]
                dev_dataset_fine_tune_id = ['corp_PRST_ar_ARG']
                test_dataset_id = ['corp_SSM_ar_ARG']

                for n in support_sizes_grid:
                    for inner_train_step in inner_train_steps_grid:
                        for inner_lr in inner_lr_grid:
                            for model in models:
                                f.write(
                                    "--bert_model {} --dev_datasets_ids {} --dev_dataset_finetune {} --test_dataset_eval {} --n {} --inner_train_steps {} --inner_lr {} --labels {}\n".format(
                                        model,
                                        ",".join(dev_dataset_ids),
                                        ",".join(dev_dataset_fine_tune_id),
                                        ",".join(test_dataset_id),
                                        n, inner_train_step, inner_lr,
                                        ",".join(all_labels_ARG)
                                    ))
    f.close()

    # Propaganda
    # (meta train) corp_PRST_aux_PTC / domain 1
    # (fine tune) corp_PRST_ar_PTC / domain 2
    # (test) corp_SSM_ar_PTC

    with open(os.path.join(save_dir, 'PTC_cross_lingual.txt'), 'w') as f:
        for lang in codes2names:
            if lang != "ar":
                dev_dataset_ids = ['corp_PRST_{}_PTC'.format(lang)]
                dev_dataset_fine_tune_id = ['corp_PRST_ar_PTC']
                test_dataset_id = ['corp_SSM_ar_PTC']

                for n in support_sizes_grid:
                    for inner_train_step in inner_train_steps_grid:
                        for inner_lr in inner_lr_grid:
                            for model in models:
                                f.write(
                                    "--bert_model {} --dev_datasets_ids {} --dev_dataset_finetune {} --test_dataset_eval {} --n {} --inner_train_steps {} --inner_lr {} --labels {}\n".format(
                                        model,
                                        ",".join(dev_dataset_ids),
                                        ",".join(dev_dataset_fine_tune_id),
                                        ",".join(test_dataset_id),
                                        n, inner_train_step, inner_lr,
                                        ";".join(all_labels_PTC)
                                    ))
    f.close()
import os


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    save_dir = "cross_domain_ar_tuning/"
    mkdir(save_dir)

    # define hyperparameters
    support_sizes_grid = [8, 16, 32]
    inner_train_steps_grid = [3, 5, 7, 10, 15]
    inner_lr_grid = [1e-4, 1e-5, 3e-4, 3e-5, 5e-4, 5e-5]
    meta_learn_iters = [30] # will fix the number of iterations (outerloop) for the sake of hyper-parameter tuning
    models = ['bert-base-multilingual-cased', 'xlm-roberta-base', 'aubmindlab/bert-base-arabertv2', 'bert-base-arabertv02', 'distilbert-base-multilingual-cased']
    modeltypes = ['bert', 'xlmroberta', 'arabert', 'arabert', 'distilbert']

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
    # (meta train) VDC_ar / domain 1
    # (meta train) corp_PRST_ar_VDC / domain 2
    # (fine tune) corp_PRST_ar_VDC / domain 2
    # (test) corp_SSM_ar_VDC

    dev_dataset_ids = ['VDC_ar', 'corp_PRST_ar_VDC']
    dev_dataset_fine_tune_id = ['corp_PRST_ar_VDC']
    test_dataset_id = ['corp_SSM_ar_VDS']
    with open(os.path.join(save_dir, 'VDC_cross_domain.txt'), 'w') as f:
        for n in support_sizes_grid:
            for inner_train_step in inner_train_steps_grid:
                for inner_lr in inner_lr_grid:
                    for m_iter in meta_learn_iters:
                        model_i = 0
                        for model in models:
                            f.write("--bert_model {} --model_type {} --dev_datasets_ids {} --dev_dataset_finetune {} --test_dataset_eval {} --do_validation 1 --do_finetuning 0 --do_evaluation 0 --meta_learn_iter {} --n {} --inner_train_steps {} --inner_lr {} --labels {} --output_dir_meta {}\n".format(
                                model, modeltypes[model_i],
                                ",".join(dev_dataset_ids),
                                ",".join(dev_dataset_fine_tune_id),
                                ",".join(test_dataset_id),
                                m_iter, n, inner_train_step, inner_lr,
                                ",".join(all_labels_VDC),
                                "results_tuning/VDC/{}-m_iter{}-inner_lr{}-inner_train_step{}-n{}/".format(model if modeltypes[model_i] != "arabert" else "_".join(model.split("/")), m_iter, inner_lr, inner_train_step, n)
                            ))
                            model_i += 1
    f.close()


    # Discourse profiling - speeches
    # (meta train) VDC_ar / domain 1
    # (meta train) corp_PRST_ar_VDS / domain 2
    # (fine tune) corp_PRST_ar_VDS / domain 2
    # (test) corp_SSM_ar_VDS
    dev_dataset_ids = ['VDC_ar', 'corp_PRST_ar_VDS']
    dev_dataset_fine_tune_id = ['corp_PRST_ar_VDS']
    test_dataset_id = ['corp_SSM_ar_VDS']
    with open(os.path.join(save_dir, 'VDS_cross_domain.txt'), 'w') as f:
        for n in support_sizes_grid:
            for inner_train_step in inner_train_steps_grid:
                for inner_lr in inner_lr_grid:
                    for m_iter in meta_learn_iters:
                        model_i = 0
                        for model in models:
                            f.write("--bert_model {} --model_type {} --dev_datasets_ids {} --dev_dataset_finetune {} --test_dataset_eval {} --do_validation 1 --do_finetuning 0 --do_evaluation 0  --meta_learn_iter {} --n {} --inner_train_steps {} --inner_lr {} --labels {} --output_dir_meta {}\n".format(
                                model, modeltypes[model_i],
                                ",".join(dev_dataset_ids),
                                ",".join(dev_dataset_fine_tune_id),
                                ",".join(test_dataset_id),
                                m_iter, n, inner_train_step, inner_lr,
                                ",".join(all_labels_VDS),
                                "results_tuning/VDS/{}-m_iter{}-inner_lr{}-inner_train_step{}-n{}/".format(model if modeltypes[model_i] != "arabert" else "_".join(model.split("/")), m_iter, inner_lr, inner_train_step, n)
                            ))

                            model_i += 1
    f.close()

    # Argumentation
    # (meta train) ARG_ar / domain 1
    # (meta train) corp_PRST_ar_ARG / domain 2
    # (fine tune) corp_PRST_ar_ARG / domain 2
    # (test) corp_SSM_ar_ARG

    # --n: number of support samples, query = batch size - n (16)
    # --inner_train_steps: number of inner updates (3)
    # --inner_lr: inner learning rate (alpha) (1e-4)
    # --learning_rate: outer learning rate (beta) (5e-5)
    # dev_datasets_ids
    # dev_dataset_finetune
    # test_dataset_eval

    dev_dataset_ids = ['ARG_ar', 'corp_PRST_ar_ARG']
    dev_dataset_fine_tune_id = ['corp_PRST_ar_ARG']
    test_dataset_id = ['corp_SSM_ar_ARG']
    with open(os.path.join(save_dir, 'argumentation_cross_domain.txt'), 'w') as f:
        for n in support_sizes_grid:
            for inner_train_step in inner_train_steps_grid:
                for inner_lr in inner_lr_grid:
                    for m_iter in meta_learn_iters:
                        model_i = 0
                        for model in models:
                            f.write("--bert_model {} --model_type {} --dev_datasets_ids {} --dev_dataset_finetune {} --test_dataset_eval {} --do_validation 1 --do_finetuning 0 --do_evaluation 0  --meta_learn_iter {} --n {} --inner_train_steps {} --inner_lr {} --labels {} --output_dir_meta {}\n".format(
                                model, modeltypes[model_i],
                                ",".join(dev_dataset_ids),
                                ",".join(dev_dataset_fine_tune_id),
                                ",".join(test_dataset_id),
                                m_iter, n, inner_train_step, inner_lr,
                                ",".join(all_labels_ARG),
                                "results_tuning/ARG/{}-m_iter{}-inner_lr{}-inner_train_step{}-n{}/".format(model if modeltypes[model_i] != "arabert" else "_".join(model.split("/")), m_iter, inner_lr, inner_train_step, n)
                            ))

                            model_i += 1
    f.close()

    # Propaganda
    # (meta train) PTC_ar / domain 1
    # (meta train) corp_PRST_ar_PTC / domain 2
    # (fine tune) corp_PRST_ar_PTC / domain 2
    # (test) corp_SSM_ar_PTC
    dev_dataset_ids = ['PTC_ar', 'corp_PRST_ar_PTC']
    dev_dataset_fine_tune_id = ['corp_PRST_ar_PTC']
    test_dataset_id = ['corp_SSM_ar_PTC']
    with open(os.path.join(save_dir, 'PTC_cross_domain.txt'), 'w') as f:
        for n in support_sizes_grid:
            for inner_train_step in inner_train_steps_grid:
                for inner_lr in inner_lr_grid:
                    for m_iter in meta_learn_iters:
                        model_i = 0
                        for model in models:
                            f.write("--bert_model {} --model_type {} --dev_datasets_ids {} --dev_dataset_finetune {} --test_dataset_eval {} --do_validation 1 --do_finetuning 0 --do_evaluation 0  --meta_learn_iter {} --n {} --inner_train_steps {} --inner_lr {} --labels {} --output_dir_meta {}\n".format(
                                model, modeltypes[model_i],
                                ",".join(dev_dataset_ids),
                                ",".join(dev_dataset_fine_tune_id),
                                ",".join(test_dataset_id),
                                m_iter, n, inner_train_step, inner_lr,
                                ";".join(all_labels_PTC),
                                "results_tuning/PTC/{}-m_iter{}-inner_lr{}-inner_train_step{}-n{}/".format(model if modeltypes[model_i] != "arabert" else "_".join(model.split("/")), m_iter, inner_lr, inner_train_step, n)
                            ))

                            model_i += 1
    f.close()
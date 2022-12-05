import json
import os
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)
import tqdm
import concurrent.futures
import pickle
import random
import torch
from transformers import AutoTokenizer
from utils.torch_utils import stack_and_pad_tensors
import glob
from collections import defaultdict
import pandas as pd
from utils.parser_utils import id2dataset_multi, id2dataset_lang, class2id, id2class, class2oheid

SUPPORT_SET_SAMPLES_KEY = "support_set_samples"
SUPPORT_SET_LENS_KEY = "support_set_lens"
TARGET_SET_SAMPLES_KEY = "target_set_samples"
TARGET_SET_LENS_KEY = "target_set_lens"
SUPPORT_SET_ENCODINGS_KEY = "support_set_encodings"
TARGET_SET_ENCODINGS_KEY = "target_set_encodings"
SELECTED_CLASS_KEY = "selected_class"
SEED_KEY = "seed"
CLASS_NAMES_KEY = "class_names"
CLASS_NAMES_ENCODING_KEY = "class_names_encoding"
CLASS_NAMES_LENS_KEY = "class_names_len"
TRAIN_DATALOADER_KEY = "train_dataloader"
DEV_DATALOADER_KEY = "dev_dataloader"


class Sample:
    def __init__(self, sentence, teacher_encoding):
        self.sentence = sentence
        self.teacher_encoding = teacher_encoding


class DistilDataLoader(DataLoader):
    def __init__(self, args):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.sample_task_to_size_ratio = args.sample_task_to_size_ratio

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights)

        self.args = args
        self.indexes_of_folders_indicating_class = (
            args.indexes_of_folders_indicating_class
        )
        self.train_val_test_split = args.train_val_test_split
        self.current_set_name = "train"
        self.proportion_intra_sample = args.proportion_intra_task_sampling
        self.variable_nr_classes_and_samples = args.variable_nr_classes_and_samples
        self.gold_label_tasks = args.gold_label_tasks
        self.gold_label_task_sample_ratio = args.gold_label_task_sample_ratio

        self.include_label_names = args.use_label_guided_learning
        self.consistency_training = args.use_consistency_loss
        self.multilingual_consistency_training = args.use_multilingual_consistency_loss

        self.num_target_samples = args.num_target_samples
        self.reset_stored_filepaths = args.reset_stored_filepaths
        val_rng = np.random.RandomState(seed=args.val_seed)
        val_seed = val_rng.randint(1, 999999)
        train_rng = np.random.RandomState(seed=args.train_seed)
        train_seed = train_rng.randint(1, 999999)
        test_rng = np.random.RandomState(seed=args.val_seed)
        test_seed = test_rng.randint(1, 999999)
        args.val_seed = val_seed
        args.train_seed = train_seed
        args.test_seed = test_seed
        self.init_seed = {
            "train": args.train_seed,
            "val": args.val_seed,
            "test": args.val_seed,
        }
        self.seed = {
            "train": args.train_seed,
            "val": args.val_seed,
            "test": args.val_seed,
        }
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.split_support_and_query = args.split_support_and_query

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.num_samples_per_class = args.num_samples_per_class
        self.num_classes_per_set = args.num_classes_per_set

        self.rng = np.random.RandomState(seed=self.seed["val"])
        self.datasets = self.load_dataset()

        self.indexes = {"train": 0, "val": 0, "test": 0}

        # Compute the number of available samples per task
        self.task_set_sizes = {
            "train": {
                label: {  # TeacherName_Lang
                    task: len(samples) for task, samples in tasks.items()
                }
                # task: index of class, samples: list of files
                for label, tasks in self.datasets["train"].items()
            },
            "val": {
                label: {  # TeacherName_Lang
                    task: len(samples) for task, samples in tasks.items()
                }
                # task: index of class, samples: list of files
                for label, tasks in self.datasets["val"].items()
            },
            "test": {
                label: {  # TeacherName_Lang
                    task: len(samples) for task, samples in tasks.items()
                }
                # task: index of class, samples: list of files
                for label, tasks in self.datasets["test"].items()
            },
        }
        # Compute the number of available samples per set
        self.class_set_sizes = {
            "train": {
                class_ix: sum([task_size for _, task_size in class_task_set.items()])
                for class_ix, class_task_set in self.task_set_sizes["train"].items()
            },
            "val": {
                class_ix: sum([task_size for _, task_size in class_task_set.items()])
                for class_ix, class_task_set in self.task_set_sizes["val"].items()
            },
            "test": {
                class_ix: sum([task_size for _, task_size in class_task_set.items()])
                for class_ix, class_task_set in self.task_set_sizes["test"].items()
            },
        }
        # # Compute the number of available samples per dataset
        self.dataset_sizes = {
            "train": sum(
                [class_size for _, class_size in self.class_set_sizes["train"].items()]
            ),
            "val": sum(
                [class_size for _, class_size in self.class_set_sizes["val"].items()]
            ),
            "test": sum(
                [class_size for _, class_size in self.class_set_sizes["test"].items()]
            ),
        }

        # self.label_set = self.get_label_set()
        self.data_length = {
            name: np.sum([len(self.datasets[name][key]) for key in self.datasets[name]])
            for name in self.datasets.keys()
        }

        print("data", self.data_length)
        self.observed_seed_set = None

        # Split available samples
        if self.split_support_and_query:
            self.split_support_and_query_sets()

    def split_support_and_query_sets(self):

        to_split = ["train"]
        if not self.args.eval_using_full_task_set:
            # to_split.extend(["val", "test"])
            to_split.extend(["val"])

        for dset in self.datasets.keys():  # for now only split the train set
            if dset not in to_split:
                continue

            for task_name in self.datasets[dset]:
                for class_name in self.datasets[dset][task_name]:
                    sample_paths = self.datasets[dset][task_name][class_name]
                    self.rng.shuffle(sample_paths)

                    support_samples = sample_paths[: len(sample_paths) // 2]
                    query_samples = sample_paths[len(sample_paths) // 2 :]

                    self.datasets[dset][task_name][class_name] = {
                        "support": support_samples,
                        "query": query_samples,
                    }

    def save_to_json(self, filename, dict_to_store):
        with open(os.path.abspath(filename), "w") as f:
            json.dump(dict_to_store, fp=f)

    def load_from_json(self, filename):
        with open(filename, mode="r") as f:
            load_dict = json.load(fp=f)

        return load_dict

    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """

        train_datasets_ids = self.args.train_datasets_ids.strip().split(",")
        dev_dataset_ids = self.args.dev_dataset_id.strip().split(",")
        test_dataset_ids = self.args.test_dataset_id.strip().split(",")

        self.index_to_label_name_dict_file = id2class
        self.label_name_to_map_dict_file = class2id

        self.splits_ids = {
            'train': train_datasets_ids,
            'val': dev_dataset_ids,
            'test': test_dataset_ids
        }
        dataset_splits = dict()
        for dset in self.splits_ids:
            dataset_splits[dset] = {}
            for id in self.splits_ids[dset]:
                lang = id2dataset_lang[id] # get language
                dataset_splits[dset][lang] = {}
                if '.xlsx' in id2dataset_multi[id]: # read the dataframe of the corresponding language
                    df = pd.read_excel(id2dataset_multi[id])
                else:
                    df = pd.read_csv(id2dataset_multi[id])

                classes = list(set(df["Label_general"]))

                for cls in classes:
                    df_cls = df[df['Label_general'] == cls] # get all rows (sentences) corresponding to a class
                    samples = []
                    for i, row in df_cls.iterrows():
                        sentence = row['Sentence']
                        sentence = ' '.join(sentence.split()[:128]) # Hiyam, let them all be mid-sized sentences
                        samples.append(Sample(sentence=sentence, teacher_encoding=class2oheid[cls]))

                    dataset_splits[dset][lang][cls] = samples

        return dataset_splits

    def load_parallel_batch(self, inputs):
        """
        Load a batch of samples, given a list of filepaths
        :param batch_sample_paths: A list of filepaths
        :return: A numpy array of samples of shape batch, height, width, channels
        """
        class_label, batch_sample_paths = inputs

        sample_batch = [
            self.load_sample(sample_path=sample_path)
            for sample_path in batch_sample_paths
        ]
        # Unzip the input ids and teacher encodings
        print(len(sample_batch))
        if len(sample_batch) == 0:
            print("empty batch")

        input_ids, teacher_encodings = zip(*sample_batch)

        return class_label, (input_ids, teacher_encodings)

    def load_batch(self, batch_samples):
        """
        Load a batch of samples, given a list of filepaths
        :param batch_sample_paths: A list of filepaths
        :return: A numpy array of samples of shape batch, height, width, channels
        """

        sample_batch = [
            self.load_sample(sample=sample_sample)
            for sample_sample in batch_samples
        ]

        # Unzip the input ids and teacher encodings
        input_ids, teacher_encodings = zip(*sample_batch)

        return (input_ids, teacher_encodings)

    def get_class_descr_samples(self, class_descr):

        x = []
        y = []
        for i, descr in enumerate(class_descr):
            # get input ids for BERT model
            input_ids = torch.LongTensor(
                self.tokenizer.encode(
                    descr.lower(),
                    text_pair=None,
                    add_special_tokens=True,
                    max_length=512,
                    truncation=True,
                )
            )
            x.append(input_ids)

            ohe_label = [0] * len(class_descr)
            ohe_label[i] = 1
            y.append(torch.FloatTensor(ohe_label))

        # Stack and pad
        x, mask = stack_and_pad_tensors(x, padding_index=self.tokenizer.pad_token_id)
        y = torch.stack(y)

        return x, mask, y

    def load_sample(self, sample):
        """
        Given an sample filepath and the number of channels to keep, load an sample and keep the specified channels
        :param sample_path: The sample's filepath
        :return: stacked and padded Tensors of input_ids and teacher_encodings
        """

        text = sample.sentence.strip()

        # get input ids for BERT model
        input_ids = torch.LongTensor(
            self.tokenizer.encode(
                text,
                text_pair=None,
                add_special_tokens=True,
                max_length=128,
                truncation=False,
            )
        )
        input_ids = input_ids[-128:]
        teacher_encodings = torch.FloatTensor([sample.teacher_encoding])

        return input_ids, teacher_encodings

    def load_test_sample(self, filepath):
        """
        Tests whether a target filepath contains a correct sample.
        :param filepath: Filepath of sample to be tested
        :return: Return filepath of sample if sample exists and is uncorrupted,
        else return None
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                sample = json.load(f)
        except json.decoder.JSONDecodeError:
            print(filepath)

        # check all necessary keys are there
        for key in [
            "source_sentence",
            "target_sentence",
            "source",
            # "target_language",
            "teacher_encoding",
            "teacher_name",
        ]:
            if key not in sample.keys():
                print("Key '{}' not found in sample {}".format(key, filepath))
                return None, None

        # check dimensions of teacher encoding
        enc = np.asarray(sample["teacher_encoding"])
        if enc.ndim != 1:
            return None, None

        try:
            float(sample["target_sentence"])
            return None, None
        except:
            pass

        return filepath, sample.get("target_language", "TEST")

    def get_full_task_set(self, task_name, percentage_train=0.8, seed=42):
        """
        Retrieves the full dataset corresponding to task_name
        :param task_name:
        :return:
        """
        rng = np.random.RandomState(seed)
        lens_by_classidx = None # won't be None when task_name is in the "test" set i.e. we are getting data loaders for "test"
        task_name_in_test = False
        # get file corresponding to task
        for d, task_mappings in self.datasets.items():
            if task_name in task_mappings.keys():

                if d == "test":
                    # pick random task from dev???
                    # get it for support
                    # get the rest of the tasks from test and make them the query for evaluation
                    lens_by_classidx = {}
                    task_name_in_test = True

                    val_task_key_name = list(self.datasets["val"].keys())[0]
                    classes_key_names = list(self.datasets["val"][val_task_key_name].keys())
                    if "support" in self.datasets["val"][val_task_key_name][classes_key_names[0]]:
                        task_sentences = [
                            self.datasets["val"][val_task_key_name][class_idx]["support"][:16] + self.datasets["val"][val_task_key_name][class_idx]["query"][:16] + task_mappings[task_name][class_idx]
                            for class_idx in task_mappings[task_name].keys()  # we are coupling the sentence with its class
                        ]

                        for class_idx in self.datasets["val"][val_task_key_name]:
                            len_supp = len(self.datasets["val"][val_task_key_name][class_idx]["support"][:16])
                            len_qry  = len(self.datasets["val"][val_task_key_name][class_idx]["query"][:16])
                            lens_by_classidx[class_idx] = len_supp + len_qry

                    else:
                        task_sentences = [
                            self.datasets["val"][val_task_key_name][class_idx][:32] + task_mappings[task_name][class_idx]
                            for class_idx in task_mappings[task_name].keys()
                            # we are coupling the sentence with its class
                        ]

                        for class_idx in self.datasets["val"][val_task_key_name]:
                            lens_by_classidx[class_idx] = len(self.datasets["val"][val_task_key_name][class_idx][:32])
                else:
                    # Tasks files are indexed per class within task
                    task_sentences = [
                        task_mappings[task_name][class_idx]
                        for class_idx in task_mappings[task_name].keys()  # we are coupling the sentence with its class
                    ]

                task_classes = []
                for class_idx in task_mappings[task_name].keys():
                    task_classes.append(class_idx)

        x_train = []
        len_train = []
        y_train = []
        x_dev = []
        len_dev = []
        y_dev = []
        label_indices = list(range(len(task_sentences)))
        # label_names = list(
        #     set([os.path.split(os.path.dirname(f[0]))[-1] for f in task_files])
        # )
        for label_ix, class_task_sentences in enumerate(task_sentences):

            if type(class_task_sentences) == dict:
                print('class_task_sentences are a dict, so will make them a list instead')
                class_task_sentences = class_task_sentences['support'] + class_task_sentences['query']

            rng.shuffle(class_task_sentences)

            num_samples = len(class_task_sentences)
            if self.args.finetune_base_model:
                num_train_samples = int(percentage_train * num_samples)
            else:
                num_bootstrap_seeds = (
                    1
                    if not self.args.bootstrap_finetune
                    else self.args.num_bootstrap_seeds
                )
                num_train_samples = (
                    self.args.num_samples_per_class * num_bootstrap_seeds
                )
            # get actual class name and index
            class_name = task_classes[label_ix] # Hiyam
            class_label = class2id[class_name] # Hiyam

            # load
            task_samples, sample_lens, task_logits = self.get_class_samples(
                class_task_sentences,
                # label_ix,
                class_label, # Hiyam
                label_indices,
                is_gold_label=self.args.val_using_cross_entropy
                or self.args.meta_loss == "ce",
                class_names=[],
                num_support_samples=0,
            )

            # print('num_train_samples: {}'.format(num_train_samples))
            # print('lens_by_classidx[class_name]: {}'.format(lens_by_classidx[class_name]))
            # print('task_logits.shape: {}'.format(task_logits.shape))
            if not task_name_in_test:
                # split
                train_set_samples = task_samples[:num_train_samples, :]
                train_set_lens = sample_lens[:num_train_samples]
                train_set_encodings = task_logits[:num_train_samples]

                dev_set_samples = task_samples[num_train_samples:, :]
                dev_set_lens = sample_lens[num_train_samples:]
                dev_set_encodings = task_logits[num_train_samples:]

                x_train.append(train_set_samples)
                len_train.append(train_set_lens)
                y_train.append(train_set_encodings)

                x_dev.append(dev_set_samples)
                len_dev.append(dev_set_lens)
                y_dev.append(dev_set_encodings)

            else:
                train_set_samples = task_samples[:lens_by_classidx[class_name], :]
                train_set_lens = sample_lens[:lens_by_classidx[class_name]]
                train_set_encodings = task_logits[:lens_by_classidx[class_name]]

                dev_set_samples = task_samples[lens_by_classidx[class_name]:, :]
                dev_set_lens = sample_lens[lens_by_classidx[class_name]:]
                dev_set_encodings = task_logits[lens_by_classidx[class_name]:]

                x_train.append(train_set_samples)
                len_train.append(train_set_lens)
                y_train.append(train_set_encodings)

                x_dev.append(dev_set_samples)
                len_dev.append(dev_set_lens)
                y_dev.append(dev_set_encodings)

        x_train = [y.squeeze() for x in x_train for y in x.split(1)]
        x_train, _ = stack_and_pad_tensors(
            x_train, padding_index=self.tokenizer.pad_token_id
        )
        len_train = torch.cat(len_train)
        # y_train = torch.cat(y_train)
        if task_name_in_test:
            y_train = torch.cat([torch.stack(y_train[i]) for i in range(len(y_train))])
            y_train = y_train.squeeze()
        else:
            y_train = torch.cat(y_train)

        x_dev = [y.squeeze() for x in x_dev for y in x.split(1)]
        len_dev = torch.cat(len_dev)
        x_dev, _ = stack_and_pad_tensors(
            x_dev, padding_index=self.tokenizer.pad_token_id
        )
        # y_dev = torch.cat(y_dev)
        if task_name_in_test:
            y_dev = torch.cat([torch.stack(y_dev[i]) for i in range(len(y_dev))])
            y_dev = y_dev.squeeze()
        else:
            y_dev = torch.cat(y_dev)

        return (
            x_train,
            len_train,
            y_train,
            x_dev,
            len_dev,
            y_dev,
            seed,
        )


    def get_num_samples_and_classes(self, num_available_classes, rng):

        if num_available_classes == 2:
            num_classes = 2
            num_samples = 8
        elif num_available_classes == 3:
            num_classes = 3
            num_samples = rng.choice([4, 5, 6])
        elif num_available_classes == 4:
            num_classes = 4
            num_samples = rng.choice([4, 5])
        elif num_available_classes == 5:
            num_classes = 5
            num_samples = rng.choice([3, 4])
        elif num_available_classes == 6:
            num_classes = 6
            num_samples = rng.choice([2, 3])
        elif num_available_classes > 6:
            num_classes = rng.choice(list(range(7, min(num_available_classes, 20))))
            if num_classes < 11:
                num_samples = 2
            else:
                num_samples = 1

        return num_classes, num_samples

    def get_set(self, dataset_name, seed):
        """
        Generates a task-set to be used for training or evaluation
        :param dataset_name: The name of the set to use, e.g. "train", "val" etc.
        :return: A task-set containing a sample and label support set, and an sample and label target set.
        """
        rng = np.random.RandomState(seed)

        sizes_class = [
            self.class_set_sizes[dataset_name][class_entry]
            for class_entry in sorted(list(self.class_set_sizes[dataset_name].keys()))
        ]

        sqrt_weights_class = np.log10(sizes_class)
        p = (
            sqrt_weights_class / np.sum(sqrt_weights_class)
            if self.sample_task_to_size_ratio
            else None
        )

        # Sample teacher/lang combination
        if rng.uniform() < self.gold_label_task_sample_ratio:
            selected_classes = rng.choice(self.gold_label_tasks, size=1, replace=False)
        else:
            selected_classes = rng.choice(
                list(self.datasets[dataset_name].keys()),
                p=p,
                size=1,
                replace=False,
            )  # Only one teacher/lang combination per set

        is_gold_label = selected_classes[0] in self.gold_label_tasks

        x_samples = []
        x_sample_lens = []
        teacher_encodings = []

        for class_entry in selected_classes:
            sizes_task = [
                self.task_set_sizes[dataset_name][class_entry][task_ix]
                for task_ix in sorted(
                            list(self.task_set_sizes[dataset_name][class_entry])
                )
            ]

            weights_task = np.log(sizes_task)
            sample_probs = (
                weights_task / np.sum(weights_task)
                if self.args.sample_task_to_size_ratio
                else None
            )
            num_sampled_labels = min(self.num_classes_per_set, len(sizes_task))

            variable_nr_samples = None
            if self.variable_nr_classes_and_samples:
                (
                    num_sampled_labels,
                    variable_nr_samples,
                ) = self.get_num_samples_and_classes(len(sizes_task), rng)

            all_label_names = list(self.datasets[dataset_name][class_entry].keys())
            selected_tasks = rng.choice(
                all_label_names,
                p=sample_probs,
                size=num_sampled_labels,
                replace=False,
            )  # Multiple classes within the teacher/lang combination

            # print('selected tasks: {} - random label idx: {}'.format(selected_tasks, list(range(len(selected_tasks)))))

            if is_gold_label or self.args.meta_loss == "ce":
                if self.args.meta_update_method.lower() == "mtl":
                    # Keep original label indexes
                    random_label_ix = [
                        all_label_names.index(label) for label in selected_tasks
                    ]
                else:
                    # random_label_ix = list(range(len(selected_tasks)))
                    random_label_ix = [class2id[c] for c in selected_tasks] # Hiyam: put that here because we are having different labels for the same class every time
            else:
                random_label_ix = [int(l) for l in selected_tasks]

            for task_entry, label_ix in zip(selected_tasks, random_label_ix):

                num_support_samples = self.num_samples_per_class
                num_query_samples = self.num_target_samples

                # Account for variability in nr of classes
                num_sample_multiplier = 1
                for i in range(1, 10):
                    if num_sampled_labels * i <= self.num_classes_per_set:
                        num_sample_multiplier = i
                    else:
                        break
                num_support_samples *= num_sample_multiplier
                num_query_samples *= num_sample_multiplier

                if variable_nr_samples is not None:
                    num_support_samples = num_query_samples = variable_nr_samples

                if self.split_support_and_query:
                    choose_samples_list = rng.choice(
                        self.datasets[dataset_name][class_entry][task_entry]["support"],
                        size=num_support_samples,
                        replace=False,
                    )
                    choose_samples_list = np.append(
                        choose_samples_list,
                        rng.choice(
                            self.datasets[dataset_name][class_entry][task_entry][
                                "query"
                            ],
                            size=num_query_samples,
                            replace=False,
                        ),
                    )

                else:
                    choose_samples_list = rng.choice(
                        self.datasets[dataset_name][class_entry][task_entry],
                        size=num_support_samples + num_query_samples,
                        replace=False,
                    )
                # Load the chosen samples
                class_samples, sample_lens, class_encodings = self.get_class_samples(
                    choose_samples_list,
                    label_ix,
                    random_label_ix,
                    is_gold_label,
                    selected_tasks,
                    num_support_samples,
                )

                x_samples.append(class_samples)
                x_sample_lens.append(sample_lens)

                class_encodings, _ = stack_and_pad_tensors(class_encodings)
                teacher_encodings.append(class_encodings)

        x_samples = [x.permute(1, 0) for x in x_samples]
        x_samples, _ = stack_and_pad_tensors(
            x_samples, padding_index=self.tokenizer.pad_token_id
        )

        x_samples = x_samples.permute(0, 2, 1)
        teacher_encodings = torch.stack(teacher_encodings)
        x_sample_lens = torch.stack(x_sample_lens)

        # x_samples: (3,32),127 , x_sample_lens: (3,32)  , teacher_encodings: (3,32),3  x_sample_lens.reshape(rows,-1).reshape(3, 32)   x_samples.reshape(rows, -1).reshape(3, 32, -1)
        # ############ Beginning of shuffling ##################
        # # I think the learning is affected by batches not being shuffled
        # # Hiyam
        rows = (num_support_samples + num_query_samples)*len(random_label_ix) # (96, 3)
        cols = len(random_label_ix) # 3

        teacher_encodings_reshaped = teacher_encodings.reshape(rows, cols)
        x_sample_lens_reshaped     = x_sample_lens.reshape(rows, -1)
        x_samples_reshaped         = x_samples.reshape(rows, -1)

        idxs = torch.randperm(teacher_encodings_reshaped.size(0)) # randomize the ordering of the classes
        teacher_encodings_reshaped = teacher_encodings_reshaped[idxs, :]
        x_sample_lens_reshaped = x_sample_lens_reshaped[idxs, :]
        x_samples_reshaped = x_samples_reshaped[idxs, :]

        teacher_encodings = teacher_encodings_reshaped.reshape(cols, num_support_samples + num_query_samples, cols)
        x_sample_lens = x_sample_lens_reshaped.reshape(cols, num_support_samples + num_query_samples)
        x_samples = x_samples_reshaped.reshape(cols, num_support_samples + num_query_samples, -1)
        # ############ End of shuffling #####################

        # Split data in support and target set
        support_set_samples = x_samples[:, :num_support_samples, :]
        support_set_lens = x_sample_lens[:, :num_support_samples]
        support_set_encodings = teacher_encodings[:, :num_support_samples]

        target_set_samples = x_samples[:, num_support_samples:, :]
        target_set_lens = x_sample_lens[:, num_support_samples:]
        target_set_encodings = teacher_encodings[:, num_support_samples:]

        res = {}

        if self.include_label_names:
            # Encode class description as substitution of prototypes
            (
                class_descr_x,
                class_descr_len,
                class_descr_y,
            ) = self.get_class_descr_samples(class_descr=selected_tasks)
            res[CLASS_NAMES_KEY] = class_descr_x
            res[CLASS_NAMES_LENS_KEY] = class_descr_len
            res[CLASS_NAMES_ENCODING_KEY] = class_descr_y

        assert (
            len(selected_classes) == 1
        ), "Only one teacher/lang combination per episode is allowed"
        selected_class = (
            selected_classes[0]
            .replace("train/", "", 1)
            .replace("val/", "", 1)
            .replace("test/", "", 1)
        )

        res.update(
            {
                SUPPORT_SET_SAMPLES_KEY: support_set_samples,
                SUPPORT_SET_LENS_KEY: support_set_lens,
                TARGET_SET_SAMPLES_KEY: target_set_samples,
                TARGET_SET_LENS_KEY: target_set_lens,
                SUPPORT_SET_ENCODINGS_KEY: support_set_encodings,
                TARGET_SET_ENCODINGS_KEY: target_set_encodings,
                SELECTED_CLASS_KEY: selected_class,
                SEED_KEY: seed,
            }
        )
        return res

    def get_class_samples(
        self,
        sample_sentences,
        label_ix,
        shuffled_labels,
        is_gold_label,
        class_names,
        num_support_samples,
    ):

        # if self.consistency_training and self.current_set_name == "train":
        #     # Add augmented samples to query set
        #     query_sample_paths = sample_paths[num_support_samples:]
        #     # Append augmented samples
        #     sample_paths = np.append(
        #         sample_paths,
        #         [
        #             self.get_aug_sample_path(sample_path)
        #             for sample_path in query_sample_paths
        #         ],
        #     )

        if self.current_set_name == "test":
            class_samples, teacher_encodings = self.load_batch(sample_sentences)
            class_samples, sample_lens = stack_and_pad_tensors(
                class_samples, padding_index=self.tokenizer.pad_token_id
            )

            return class_samples, sample_lens, teacher_encodings

        else:
            # Loads and prepares samples for 1 class within task
            class_samples, teacher_encodings = self.load_batch(sample_sentences)

            class_samples, sample_lens = stack_and_pad_tensors(
                class_samples, padding_index=self.tokenizer.pad_token_id
            )

            if self.args.meta_update_method == "mtl":
                class_encodings = torch.stack(teacher_encodings)
            elif self.args.meta_loss == "ce" or is_gold_label:
                ohe_label = [0] * len(shuffled_labels)
                ohe_label[label_ix] = 1
                class_encodings = torch.LongTensor([ohe_label] * len(sample_sentences))
            else:  # kl
                # Index the teacher logits at indices of target classes
                class_encodings = torch.stack(teacher_encodings)[:, shuffled_labels]

            return class_samples, sample_lens, class_encodings

    def get_aug_sample_path(self, sample_path):
        sample_dir, sample_name = os.path.split(sample_path)

        sample_name = sample_name.replace(".json", "").replace("Train", "")

        aug_sample_dir = sample_dir.replace("/train/", "/aug/").replace("Train", "")
        if self.multilingual_consistency_training:
            aug_sample_dir, class_name = os.path.split(aug_sample_dir)
            aug_sample_dir = os.path.dirname(aug_sample_dir)
            available_langs = os.listdir(aug_sample_dir)
            aug_sample_dir = os.path.join(
                aug_sample_dir, self.rng.choice(available_langs, size=1)[0], class_name
            )

        aug_candidates = [
            os.path.join(aug_sample_dir, f)
            for f in os.listdir(aug_sample_dir)
            if f.startswith(sample_name)
        ]

        return self.rng.choice(aug_candidates, size=1)

    def __len__(self):
        total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name
        if set_name == "train":
            self.update_seed(
                dataset_name=set_name, seed=self.init_seed[set_name] + current_iter
            )

    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx):
        return self.get_set(
                self.current_set_name, seed=self.seed[self.current_set_name] + idx
        )

    def reset_seed(self):
        self.seed = self.init_seed


def collate_fn(batch):
    # Dissect batch
    support_set_samples = [b[SUPPORT_SET_SAMPLES_KEY] for b in batch]
    support_set_lens = [b[SUPPORT_SET_LENS_KEY] for b in batch]
    target_set_samples = [b[TARGET_SET_SAMPLES_KEY] for b in batch]
    target_set_lens = [b[TARGET_SET_LENS_KEY] for b in batch]
    support_set_labels = [b[SUPPORT_SET_ENCODINGS_KEY] for b in batch]
    target_set_labels = [b[TARGET_SET_ENCODINGS_KEY] for b in batch]
    selected_classes = [b[SELECTED_CLASS_KEY] for b in batch]
    seeds = [b[SEED_KEY] for b in batch]

    res = {}
    if CLASS_NAMES_KEY in batch[0].keys():
        # Process class description samples
        class_names_x = [b[CLASS_NAMES_KEY] for b in batch]
        class_names_len = [b[CLASS_NAMES_LENS_KEY] for b in batch]
        class_names_y = [b[CLASS_NAMES_ENCODING_KEY] for b in batch]

        class_names_mask = [torch.ones_like(s) for s in class_names_x]
        class_names_mask = [
            (torch.arange(s.size(1)) < l.contiguous().view(-1).unsqueeze(1)) * s
            for s, l in zip(class_names_mask, class_names_len)
        ]
        # Add to final dict
        res[CLASS_NAMES_KEY] = class_names_x
        res[CLASS_NAMES_LENS_KEY] = class_names_mask
        res[CLASS_NAMES_ENCODING_KEY] = class_names_y

    # Flatten samples
    support_set_samples = [
        s.contiguous().view(s.size(0) * s.size(1), -1) for s in support_set_samples
    ]

    target_set_samples = [
        s.contiguous().view(s.size(0) * s.size(1), -1) for s in target_set_samples
    ]

    # Get attention masks from original lengths of sequence
    support_set_mask = [torch.ones_like(s) for s in support_set_samples]
    support_set_mask = [
        (torch.arange(s.size(1)) < l.contiguous().view(-1).unsqueeze(1)) * s
        for s, l in zip(support_set_mask, support_set_lens)
    ]

    target_set_mask = [torch.ones_like(s) for s in target_set_samples]
    target_set_mask = [
        (torch.arange(s.size(1)) < l.contiguous().view(-1).unsqueeze(1)) * s
        for s, l in zip(target_set_mask, target_set_lens)
    ]

    # Flatten targets
    support_set_labels = [
        s.contiguous().view(s.size(0) * s.size(1), -1) for s in support_set_labels
    ]
    target_set_labels = [
        s.contiguous().view(s.size(0) * s.size(1), -1) for s in target_set_labels
    ]

    res.update(
        {
            SUPPORT_SET_SAMPLES_KEY: support_set_samples,
            SUPPORT_SET_LENS_KEY: support_set_mask,
            TARGET_SET_SAMPLES_KEY: target_set_samples,
            TARGET_SET_LENS_KEY: target_set_mask,
            SUPPORT_SET_ENCODINGS_KEY: support_set_labels,
            TARGET_SET_ENCODINGS_KEY: target_set_labels,
            SELECTED_CLASS_KEY: selected_classes,
            SEED_KEY: seeds,
        }
    )

    return res


class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.samples_per_iter = args.samples_per_iter
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.dataset = DistilDataLoader(args=args)
        self.batches_per_iter = args.samples_per_iter
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)
        self.args = args

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """

        return DataLoader(
            self.dataset,
            batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )


    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += current_iter * (
            self.num_of_gpus * self.batch_size * self.samples_per_iter
        )

    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_samples: Whether we want the samples to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(
            set_name="train", current_iter=self.total_train_iters_produced
        )

        self.total_train_iters_produced += (
            self.num_of_gpus * self.batch_size * self.samples_per_iter
        )
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_val_batches(self, total_batches=-1):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_samples: Whether we want the samples to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["val"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")

        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_test_batches(self, total_batches=-1):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_samples: Whether we want the samples to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["test"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="test")

        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_finetune_dataloaders(self, task_name, percentage_train, seed):

        # self.dataset.switch_set(set_name="val")  # TODO: tmp
        (
            train_set_samples,
            train_set_lens,
            train_set_encodings,
            dev_set_samples,
            dev_set_lens,
            dev_set_encodings,
            seed,
        ) = self.dataset.get_full_task_set(task_name, percentage_train, seed)

        train_mask = torch.ones_like(train_set_samples)
        train_mask = (
            torch.arange(train_mask.size(1))
            < train_set_lens.contiguous().view(-1).unsqueeze(1)
        ) * train_mask

        train_dataset = TensorDataset(
            train_set_samples, train_mask, train_set_encodings
        )
        train_sampler = SequentialSampler(train_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.dataset.num_samples_per_class
            * self.dataset.num_classes_per_set,
        )

        dev_mask = torch.ones_like(dev_set_samples)
        dev_mask = (
            torch.arange(dev_mask.size(1))
            < dev_set_lens.contiguous().view(-1).unsqueeze(1)
        ) * dev_mask

        dev_dataset = TensorDataset(dev_set_samples, dev_mask, dev_set_encodings)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(
            dev_dataset,
            sampler=dev_sampler,
            batch_size=self.dataset.num_samples_per_class
            * self.dataset.num_classes_per_set
            * 5,
        )
        res = {}
        if self.dataset.include_label_names:
            class_names = ""
            for d, task_mappings in self.dataset.datasets.items():
                if task_name in task_mappings.keys():
                    class_names = list(task_mappings[task_name].keys())

            (
                class_descr_x,
                class_descr_len,
                class_descr_y,
            ) = self.dataset.get_class_descr_samples(class_descr=class_names)
            res[CLASS_NAMES_KEY] = class_descr_x
            res[CLASS_NAMES_LENS_KEY] = torch.ones_like(class_descr_x)
            res[CLASS_NAMES_ENCODING_KEY] = class_descr_y

        res.update(
            {TRAIN_DATALOADER_KEY: train_dataloader, DEV_DATALOADER_KEY: dev_dataloader}
        )

        return res
# coding: UTF-8
import torch.nn as nn
import time
import torch
import numpy as np
import argparse
from utils import build_dataset, build_iterator
import random
from tqdm import tqdm, trange
from copy import deepcopy
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix, accuracy_score
import logging
import os

logging.basicConfig(format='%(message)s', #"format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Config(object):

    """ Configuration parameters """
    # def __init__(self, dataset, embedding):
    def __init__(self, args):
        self.model_name = args.model
        self.train_path = args.train_set
        self.dev_path = args.dev_set
        self.test_path = args.test_set
        self.text_column = args.text_column
        self.label_column = args.label_column
        self.save_path = 'saved_dict/'
        self.log_path = 'log/' + args.model + '/'
        self.embedding_path = args.embedding_path
        self.vocab_path = args.vocab_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = args.dropout # 0.5
        self.require_improvement = args.require_improvement # 1000
        self.class_list = None
        self.num_classes = None
        self.n_vocab = 0
        self.num_epochs = args.num_epochs # 1000
        self.batch_size = args.batch_size # 32
        # self.pad_size = args.pad_size # 32
        self.max_sen_len = args.max_sen_len
        self.learning_rate = args.lr # 1e-3

        if args.fasttext:
            self.embedding_pretrained = None
            self.embed = 300
        else:
            self.embedding_pretrained = torch.tensor(np.load(self.embedding_path)["embeddings"].astype('float32'))
            self.embed = self.embedding_pretrained.size(1)

        self.hidden_size = args.hidden_size # 128
        self.num_layers = args.num_layers # 2

        if self.model_name == 'TextCNN':
            self.filter_sizes = (2, 3, 4)
            self.num_filters = 256
        else:
            self.filter_sizes = None
            self.num_filters = None

        if self.model_name == 'TextRNN_Att':
            self.hidden_size2 = 64
        else:
            self.hidden_size2 = None
        self.pad_size = None


class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x.long())  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out


class TextRNN_Att(nn.Module):
    def __init__(self, config):
        super(TextRNN_Att, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        # x, _ = x
        # emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        emb = self.embedding(x.long())  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out


class TextRCNN(nn.Module):
    def __init__(self, config):
        super(TextRCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        # x, _ = x
        # embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        embed = self.embedding(x.long())  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # out = self.embedding(x[0])
        out = self.embedding(x.long())
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class DPCNN(nn.Module):
    def __init__(self, config):
        super(DPCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        # x = x[0]
        x = x.long()
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


def _sample_mini_dataset(dataset, num_classes, num_shots):
    random.shuffle(dataset)
    for cls in range(num_classes):
        cls_indices = [i for i, sentence_content in enumerate(dataset) if sentence_content[1] == cls]  # get indices of all examples pertaining to the class cls
        selected = np.random.choice(cls_indices, num_shots, replace=False)  # randomly select num_shots number of indices
        for i in selected:
            yield dataset[i]


def _mini_batches(samples, batch_size, num_batches, replacement):
    """
        Generate mini-batches from some data.
        :param samples: a mini-dataset containing num_shots*num_classes number of examples (per class)
        :param batch_size: size of a single batch
        :param num_batches: number of batches to generate
        :param replacement: boolean: whetehr to sample examples w/o replacement
        :return:
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return


def _to_tensor(datas, device):
    x = torch.LongTensor([_[0] for _ in datas]).to(device)
    y = torch.LongTensor([_[1] for _ in datas]).to(device)

    return x, y


def compute_metrics(preds, labels):
    # check if binary or multiclass classification
    if len(list(set(labels))) == 2:
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
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
        return {
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc_and_f1": (accuracy + f1) / 2,
        }


def mkdir(folder):
    """ create the directory, if it doesn't already exist """
    if not os.path.exists(folder):
        os.makedirs(folder)


def main():
    parser = argparse.ArgumentParser(description='Arabic Text Classification')


    parser.add_argument('--model', type=str, default='TextCNN',
                        help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')


    parser.add_argument('--train_set', type=str, default='input/FAKES/feature_extraction_train_updated_updated.csv',
                        help='path to training dataset')
    parser.add_argument('--dev_set', type=str, default='input/FAKES/feature_extraction_dev_updated_updated.csv',
                        help='path to validation dataset')
    parser.add_argument('--test_set', type=str, default='input/FAKES/feature_extraction_test_updated.csv',
                        help='path to testing dataset')
    parser.add_argument('--text_column', type=str, default='article_content',
                        help='name of the col containing text data inside train/val/test files')
    parser.add_argument('--label_column', type=str, default='label',
                        help='name of the col containing labels inside train/val/test files')
    parser.add_argument('--embedding_path', default='embeddings/fakes_embeddings.npz', type=str,
                        help='random or path to pre_trained embedding')
    parser.add_argument('--vocab_path', default='vocab/fakes_vocab.pkl', type=str,
                        help='path to the vocabulary mapping each word to an index')
    parser.add_argument('--fasttext', default=0, type=int,
                        help='whether to use fasttext embeddings or not. 0=False, 1=True')
    parser.add_argument('--max_sen_len', type=int, default=512, help='maximum length of sentence')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--require_improvement', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--pad_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=128, help='number of nodes in hidden layer')
    parser.add_argument('--num_layers', type=int, default=2, help='numer of hidden layers')
    parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')

    # FOMAML arguments
    parser.add_argument('--inner_iters', type=int, default=20, help='inner iterations')
    parser.add_argument('--inner_batch', type=int, default=5, help='inner batch size')
    parser.add_argument('--num_shots', type=int, default=5, help='number of examples per class')
    parser.add_argument('--meta_batch', type=int, default=5, help='meta training batch size')
    parser.add_argument('--meta_iters', type=int, default=1, help='meta-training iterations')
    parser.add_argument("--is_reptile", default=False, type=bool, help="whether use reptile or fomaml method")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--inner_learning_rate", default=2e-6, type=float, help="The inner learning rate for Adam")
    parser.add_argument("--outer_learning_rate", default=1e-5, type=float, help="The meta learning rate for Adam, actual learning rate!")
    parser.add_argument("--FSL_learning_rate", default=2e-5, type=float, help="The FSL learning rate for Adam!")
    parser.add_argument("--FSL_epochs", default=1, type=int, help="The FSL learning epochs for training!")
    parser.add_argument("--do_train", action='store_false', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_false', help="Whether to run eval on the dev set.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    print('running FOMAML with {} as a base model ...'.format(model_name))

    config = Config(args)

    np.random.seed(1)
    torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data, class_list = build_dataset(config, args.word)
    config.class_list = class_list
    config.num_classes = len(class_list)

    inner_iters = args.inner_iters  # number of inner iterations
    meta_iters = args.meta_iters  # number of meta train iterations
    num_shots = args.num_shots  # number of examples per class in the support set
    meta_batch = args.meta_batch  # meta_batch_size: number of tasks to sample **per meta iteration**
    N_class = config.num_classes  # number of classes in each task
    inner_batch = args.inner_batch  # ??
    Is_reptile = args.is_reptile

    num_train_optimization_steps = meta_iters * (num_shots) * N_class * meta_batch
    if model_name == 'TextRNN':
        model = TextRNN(config).to(config.device)
    elif model_name == 'TextRNN_Att':
        model = TextRNN_Att(config).to(config.device)
    elif model_name == 'TextRCNN':
        model = TextRCNN(config).to(config.device)
    elif model_name == 'TextCNN':
        model = TextCNN(config).to(config.device)
    else:
        model = DPCNN(config).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if args.do_train:
        for epoch in trange(int(meta_iters), desc="Iterations"):  # meta-training
            weight_before = deepcopy(model.state_dict())
            update_vars = []
            fomaml_vars = []

            for _ in range(meta_batch):  # N_task is meta_batch_size

                # here, we sample a mini-dataset, from which we sample examples per class in the inner loop
                # https://github.com/openai/supervised-reptile/blob/8f2b71c67a31c1a605ced0cecb76db876b607a7a/supervised_reptile/reptile.py#L65

                mini_dataset = _sample_mini_dataset(dataset=train_data, num_classes=N_class, num_shots=num_shots)

                for batch in _mini_batches(samples=mini_dataset, batch_size=inner_batch, num_batches=inner_iters, replacement=False):  # for inner number of updates -- this is the inner loop over mini batches
                    # https://github.com/openai/supervised-reptile/blob/8f2b71c67a31c1a605ced0cecb76db876b607a7a/supervised_reptile/reptile.py#L66

                    inputs, labels = _to_tensor(datas=batch, device=config.device)

                    last_backup = deepcopy(model.state_dict())

                    model.train()

                    outputs = model(inputs.float())
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if epoch % 10 == 0:
                        true = labels.data.cpu()
                        predic = torch.max(outputs.data, 1)[1].cpu()

                        result = compute_metrics(preds=predic, labels=true)
                        task_acc = result['acc']
                        task_f1 = result['f1']
                        task_prec = result['precision']
                        task_rec = result['recall']
                        print('epoch: {}, accuracy: {}, precision: {}, recall: {}, f1: {}, loss: {}'.format(epoch, task_acc,
                                                                                                            task_prec,
                                                                                                            task_rec,
                                                                                                            task_f1, loss))

                weight_after = deepcopy(model.state_dict())
                update_vars.append(weight_after)
                tmp_fomaml_var = {}

                if not Is_reptile:
                    for name in weight_after:
                        tmp_fomaml_var[name] = weight_after[name] - last_backup[name]
                    fomaml_vars.append(tmp_fomaml_var)
                model.load_state_dict(
                    weight_before)  # here we are: https://github.com/openai/supervised-reptile/blob/8f2b71c67a31c1a605ced0cecb76db876b607a7a/supervised_reptile/reptile.py#L72

            new_weight_dict = {}
            # print(weight_before)
            if Is_reptile:
                for name in weight_before:
                    weight_list = [tmp_weight_dict[name] for tmp_weight_dict in update_vars]
                    weight_shape = list(weight_list[0].size())
                    stack_shape = [len(weight_list)] + weight_shape
                    stack_weight = torch.empty(stack_shape)
                    for i in range(len(weight_list)):
                        stack_weight[i, :] = weight_list[i]
                    if device == "gpu":
                        new_weight_dict[name] = torch.mean(stack_weight, dim=0).cuda()
                    else:
                        new_weight_dict[name] = torch.mean(stack_weight, dim=0)
                    # new_weight_dict[name] = torch.mean(stack_weight, dim=0)
                    new_weight_dict[name] = weight_before[name].to(device) + (new_weight_dict[name].to(device) - weight_before[name].to(device)) / args.inner_learning_rate * args.outer_learning_rate
            else:
                for name in weight_before:
                    weight_list = [tmp_weight_dict[name] for tmp_weight_dict in fomaml_vars]
                    weight_shape = list(weight_list[0].size())
                    stack_shape = [len(weight_list)] + weight_shape
                    stack_weight = torch.empty(stack_shape)
                    for i in range(len(weight_list)):
                        stack_weight[i, :] = weight_list[i]
                    if device == "gpu":
                        new_weight_dict[name] = torch.mean(stack_weight, dim=0).cuda()
                    else:
                        new_weight_dict[name] = torch.mean(stack_weight, dim=0)
                    # new_weight_dict[name] = torch.mean(stack_weight, dim=0).cuda()
                    # new_weight_dict[name] = torch.mean(stack_weight, dim=0)
                    new_weight_dict[name] = weight_before[name].to(device) + new_weight_dict[name].to(device) / args.inner_learning_rate * args.outer_learning_rate
            model.load_state_dict(new_weight_dict)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) :
        mkdir(config.save_path)
        torch.save(model.state_dict(), os.path.join(config.save_path, '{}.ckpt'.format(config.model_name)))

    model.to(device)

    Meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.FSL_learning_rate)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.do_train:
            weight_before = deepcopy(model.state_dict())
            model.train()
        else:
            model = RNNModel(config).to(config.device)
            model.load_state_dict(torch.load(config.save_path + config.model_name + '.ckpt'))
            weight_before = deepcopy(model.state_dict())

            model.train()

        mini_dataset = _sample_mini_dataset(dataset=dev_data, num_classes=N_class, num_shots=num_shots)

        for batch in _mini_batches(samples=mini_dataset, batch_size=inner_batch, num_batches=inner_iters, replacement=False):

            inputs, labels = _to_tensor(datas=batch, device=config.device)

            outputs = model(inputs.float())
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()  # calculate gradients and update weights
            Meta_optimizer.step()
            Meta_optimizer.zero_grad()

        model.eval()
        eval_dataloader = build_iterator(dataset=test_data, config=config)

        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in eval_dataloader:
                outputs = model(texts)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        result = compute_metrics(preds=predict_all, labels=labels_all)
        report = classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = confusion_matrix(labels_all, predict_all)

        print('Results:')
        for k, v in result.items():
            print('{}: {:.5f}'.format(k, v))
        print('\nClassification report:\n{}'.format(report))
        print('\nConfusion matrix:\n{}'.format(confusion))

if __name__ == '__main__':
    main()
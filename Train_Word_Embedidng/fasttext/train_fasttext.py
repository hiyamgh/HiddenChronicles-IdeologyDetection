import fasttext
import argparse
import os, logging
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue


def mkdir(folder):
    ''' creates a directory if it doesn't already exist '''
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--archive', type=str, default='assafir', help="name of the archive to transform")
    parser.add_argument('-c', '--minCount', type=int, help='minimal number of word occurrences')
    parser.add_argument('-w', '--wordNgrams', type=int, help='max length of word ngram')
    parser.add_argument('-m', '--model', type=str, help='type of model - cbow or skipgram')
    parser.add_argument('-l', '--lr', type=float, help='learning rate')
    parser.add_argument('-d', '--dim', type=int, help='size of word vectors')
    parser.add_argument('-s', '--ws', type=int, help='size of the context window')
    parser.add_argument('-n', '--neg', type=int, help='number of negatives sampled')
    parser.add_argument('-y', '--year', type=int, help='year to train on')
    args = parser.parse_args()

    archive = args.archive
    data_folder = "data/{}/".format(archive)
    # the output folder to save trained model in
    logdir = '{}/{}/ngrams{}-size{}-window{}-mincount{}-negative{}-lr{}/'.format(args.archive,
                                                                             'cbow' if args.model == 'cbow' else 'SGNS',
                                                                             args.year,
                                                                             args.wordNgrams,
                                                                             # 'cbow' if args.model == 'cbow' else 'SGNS',
                                                                             args.dim, args.ws, args.minCount,
                                                                             args.neg if args.model == 'skipgram' else 0,
                                                                             args.lr)

    mkdir(logdir)
    input_file = '{}.txt'.format(args.year)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print('\nTraining fasttext on {} archive for year {}]\n'.format(args.archive, args.year))
    model = fasttext.train_unsupervised(input=os.path.join(data_folder, input_file),
                                        wordNgrams=args.wordNgrams,
                                        model=args.model,
                                        lr=args.lr,
                                        dim=args.dim,
                                        ws=args.ws,
                                        neg=args.neg if args.model == 'skipgram' else 0,
                                        thread=mp.cpu_count())

    model.save_model(os.path.join(logdir, "{}.bin".format(args.year)))


    # for input_file in os.listdir(data_folder):
    #     year = input_file.split('.')[0]
    #     print('year is: {}\n'.format(year))
    #     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #     model = fasttext.train_unsupervised(input=os.path.join(data_folder, input_file),
    #                                 wordNgrams=args.wordNgrams,
    #                                 model=args.model,
    #                                 lr=args.lr,
    #                                 dim=args.dim,
    #                                 ws=args.ws,
    #                                 neg=args.neg if args.model == 'skipgram' else 0,
    #                                 thread=mp.cpu_count())
    #
    #     model.save_model(os.path.join(logdir, "{}.bin".format(year)))




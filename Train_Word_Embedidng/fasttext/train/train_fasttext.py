import fasttext
import argparse
import os, logging
import multiprocessing as mp


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
    parser.add_argument('-y', '--year', type=int, help='year to train on') # use if training at the yearly level
    parser.add_argument('-sy', '--start_year', type=int, default=None, help='start year to include') # use if training several years to one embedding
    parser.add_argument('-ey', '--end_year', type=int, default=None, help='end year to include') # use if tarining several years to one embedding
    args = parser.parse_args()

    archive = args.archive

    if args.start_year is None and args.end_year is None:
        data_folder = "data/{}/".format(archive)
        input_file = '{}.txt'.format(args.year)
        logdir = '{}/{}/ngrams{}-size{}-window{}-mincount{}-negative{}-lr{}/'.format(args.archive,
                                                                                     'cbow' if args.model == 'cbow' else 'SGNS',
                                                                                     args.wordNgrams,
                                                                                     # 'cbow' if args.model == 'cbow' else 'SGNS',
                                                                                     args.dim, args.ws, args.minCount,
                                                                                     args.neg if args.model == 'skipgram' else 0,
                                                                                     args.lr)

    else:
        data_folder = "data/{}/start_end/".format(archive)
        input_file = '{}-{}.txt'.format(args.start_year, args.end_year)
        logdir = '{}/{}_{}/{}/ngrams{}-size{}-window{}-mincount{}-negative{}-lr{}/'.format(args.archive, args.start_year, args.end_year,
                                                                                           'cbow' if args.model == 'cbow' else 'SGNS',
                                                                                           args.wordNgrams,
                                                                                           # 'cbow' if args.model == 'cbow' else 'SGNS',
                                                                                           args.dim, args.ws,
                                                                                           args.minCount,
                                                                                           args.neg if args.model == 'skipgram' else 0,
                                                                                           args.lr)

    if os.path.exists(os.path.join(data_folder, input_file)):

        if args.start_year is None and args.end_year is None:
            print('started training fasttext word embedding on {} archive, year = {}'.format(archive, args.year))
        else:
            print('started training fasttext word embedding on {} archive, years = {}-{}'.format(archive, args.start_year, args.end_year))

        mkdir(logdir)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        model = fasttext.train_unsupervised(input=os.path.join(data_folder, input_file),
                                            wordNgrams=args.wordNgrams,
                                            model=args.model,
                                            lr=args.lr,
                                            dim=args.dim,
                                            ws=args.ws,
                                            neg=args.neg if args.model == 'skipgram' else 0,
                                            thread=mp.cpu_count())

        if args.start_year is None and args.end_year is None:
            model.save_model(os.path.join(logdir, "{}.bin".format(args.year)))
        else:
            model.save_model(os.path.join(logdir, "{}_{}.bin".format(args.start_year, args.end_year)))









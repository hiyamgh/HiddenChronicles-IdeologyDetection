import fasttext
import argparse
import os, logging

def mkdir(folder):
    ''' creates a directory if it doesn't already exist '''
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--archive', type=str, default='nahar', help="name of the archive to transform")
    parser.add_argument('-y', '--year', type=int, help='year to train on') # use if training at the yearly level
    args = parser.parse_args()

    archive = args.archive

    data_folder = "data/{}/".format(archive)
    input_file = '{}.txt'.format(args.year)
    logdir = 'trained_models/{}-skipgram-neg15/'.format(archive)

    if os.path.exists(os.path.join(data_folder, input_file)):
        print('started training fasttext word embedding on {} archive, year = {}'.format(archive, args.year))
        mkdir(logdir)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        model = fasttext.train_unsupervised(input=os.path.join(data_folder, input_file), model='skipgram', neg=15)
        model.save_model(os.path.join(logdir, "{}.bin".format(args.year)))










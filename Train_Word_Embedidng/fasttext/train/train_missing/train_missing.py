import fasttext
import argparse
import os, logging


def mkdir(folder):
    ''' creates a directory if it doesn't already exist '''
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=str, default='nahar', help="name of the archive to transform")
    parser.add_argument('--start_year', type=int, default=1933, help='The first year present in the archive')
    parser.add_argument('--end_year', type=int, default=2009, help='The first year present in the archive')
    args = parser.parse_args()

    archive = args.archive

    data_folder = "data/{}/".format(archive)
    # input_file = '{}.txt'.format(args.year)
    logdir = 'trained_models/{}-skipgram-neg15/'.format(archive)
    start_year = args.start_year
    end_year = args.end_year
    if archive != 'hayat':
        list_of_files = ['{}.bin'.format(y) for y in range(start_year, end_year + 1)]
    else:
        # 1950 till 1976, and from 1988 till 2000
        years = [y for y in range(start_year, 1977)] + [y for y in range(1988, end_year + 1)]
        list_of_files = ['{}.bin'.format(y) for y in years]

    # if os.path.exists(os.path.join(data_folder, input_file)):
    input_files = []
    for file in list_of_files:
        if not os.path.isfile(os.path.join(logdir, file)):
            input_file = '{}.txt'.format(file[:-4])
            input_files.append(input_file)

    if input_files != []:
        for input_file in input_files:
            print('started training fasttext word embedding on {} archive, year = {}'.format(archive, input_file[:-4]))
            mkdir(logdir)

            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

            model = fasttext.train_unsupervised(input=os.path.join(data_folder, input_file), model='skipgram', neg=15)
            model.save_model(os.path.join(logdir, "{}.bin".format(input_file[:-4])))
    else:
        print('all files are found in {}'.format(logdir))










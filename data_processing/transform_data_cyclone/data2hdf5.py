'''
Gather all the raw txt files into a tree like format (h5py) in order to access
certain txt by year/month/date etc.
'''

import os
import h5py
import argparse

# https://docs.h5py.org/en/stable/special.html


def mkdir(directory):
    ''' creates a directory os specified path if it does not exist already '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_text_dirs():
    '''
        intended to construct a dictionary of the path to different
        batches of data for each arhcive we have
    '''
    # nahar archive
    TEXT_DIR1_nahar = '../../../../nahar/nahar-batch-1/out/'
    TEXT_DIR2_nahar = '../../../../nahar/nahar-batch-2/out/'
    TEXT_DIR3_nahar = '../../../../nahar/nahar-batch-3/out/'
    TEXT_DIR4_nahar = '../../../../nahar/nahar-batch-4/out/'

    # assafir archive
    TEXT_DIR1_assafir = '../../../../assafir/assafir-batch-1/out/'
    TEXT_DIR2_assafir = '../../../../assafir/assafir-batch-2/out/'

    # hayat archive
    TEXT_DIR1_hayat = '../../../../hayat/hayat-batch-1/out/'
    TEXT_DIR2_hayat = '../../../../hayat/hayat-batch-2/out/'

    # txt files directory for nahar
    TEXT_DIRS_nahar = [TEXT_DIR1_nahar, TEXT_DIR2_nahar, TEXT_DIR3_nahar, TEXT_DIR4_nahar]
    TEXT_DIRS_assafir = [TEXT_DIR1_assafir, TEXT_DIR2_assafir]
    TEXT_DIRS_hayat = [TEXT_DIR1_hayat, TEXT_DIR2_hayat]

    newspapers_dict = {
        'nahar': TEXT_DIRS_nahar,
        'assafir': TEXT_DIRS_assafir,
        'hayat': TEXT_DIRS_hayat,
    }

    return newspapers_dict


def create_archive_hdf5(TEXT_DIRS, archive, output_folder):
    ''' create text dirs for a certain archive '''
    mkdir(output_folder)
    hf = h5py.File(os.path.join(output_folder, '{}.h5').format(archive), 'w')
    for txt_dir in TEXT_DIRS:
        count = 0
        print('processing files in {}'.format(txt_dir))
        for issuepage in os.listdir(txt_dir):
            if issuepage[0].isdigit() and issuepage.endswith('.txt'):
                year = issuepage[:2]
                month = issuepage[2:4]
                day = issuepage[4:6]
                pagenb = issuepage[6:8]

                # create groups by years numbers, if group already exists append to the group
                if int(year) < 20:
                    year_str = '20{}'.format(year)
                    if year_str not in hf.keys():
                        groupID = hf.create_group(year_str)
                    else:
                        groupID = hf[year_str]
                else:
                    year_str = '19{}'.format(year)
                    if year_str not in hf.keys():
                        groupID = hf.create_group(year_str)
                    else:
                        groupID = hf[year_str]

                text_file = open(os.path.join(txt_dir, issuepage), encoding='utf-8')
                string_dt = h5py.special_dtype(vlen=str)
                # create dataset with raw data
                dataset = groupID.create_dataset('{}'.format(issuepage), dtype=string_dt, data=text_file.read())
                # create dataset with cleaned data
                # lines = text_file.readlines()
                # lines_cleaned = arabnormalizer.normalize_paragraph(lines)
                # str_clean = ''
                # for line in lines_cleaned:
                #     if line == '\n':
                #         str_clean += line
                #     else:
                #         str_clean += line + '\n'
                # dset_clean = groupID.create_dataset('{}-clean'.format(issuepage), dtype=string_dt, data=str_clean)
                # add attributes about the issue
                dataset.attrs['year'] = year_str
                dataset.attrs['month'] = month
                dataset.attrs['day'] = day
                dataset.attrs['pagenb'] = pagenb

                count += 1

                if count % 1000 == 0:
                    print('processed {} files so far'.format(count))

    hf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=str, help="name of the archive to transform")
    args = parser.parse_args()

    newspapers_dict = get_text_dirs()
    TEXT_DIRS_archive = newspapers_dict[args.archive]
    create_archive_hdf5(TEXT_DIRS_archive, archive=args.archive, output_folder='../../')

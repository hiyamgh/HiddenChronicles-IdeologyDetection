import os
import h5py
from normalization import ArabicNormalizer
''' 
Gather all the raw txt files into a tree like format (h5py) in order to access
certain txt by year/month/date etc.
'''

# https://docs.h5py.org/en/stable/special.html


def get_text_dirs():
    # nahar archive
    TEXT_DIR1_nahar = '../../input/nahar/nahar-batch-1/out/'
    TEXT_DIR2_nahar = '../../input/nahar/nahar-batch-2/out/'
    TEXT_DIR3_nahar = '../../input/nahar/nahar-batch-3/out/'
    TEXT_DIR4_nahar = '../../input/nahar/nahar-batch-4/out/'

    # assafir archive
    TEXT_DIR1_assafir = '../../input/assafir/assafir-batch-1/out/'
    TEXT_DIR2_assafir = '../../input/assafir/assafir-batch-2/out/'

    # hayat archive
    TEXT_DIR1_hayat = '../../input/hayat/hayat-batch-1/out/'
    TEXT_DIR2_hayat = '../../input/hayat/hayat-batch-2/out/'

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


def create_archive_hdf5(TEXT_DIRS, archive):
    ''' create text dirs for a certain archive '''
    hf = h5py.File('{}.h5'.format(archive), 'w')
    arabnormalizer = ArabicNormalizer()
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
                groupID.create_dataset('text-raw', dtype=string_dt, data=text_file.readlines())
                # create dataset with cleaned data
                lines = text_file.readlines()
                lines_cleaned = arabnormalizer.normalize_paragraph(lines)
                str_clean = ''
                for line in lines_cleaned:
                    if line == '\n':
                        str_clean += line
                    else:
                        str_clean += line + '\n'
                dset_clean = groupID.create_dataset('text-clean', dtype=string_dt, data=str_clean)
                # add attributes about the issue
                dset_clean.attrs['year'] = year_str
                dset_clean.attrs['month'] = month
                dset_clean.attrs['day'] = day
                dset_clean.attrs['pagenb'] = pagenb

                count += 1

                if count % 1000 == 0:
                    print('processed {} files so far'.format(count))

    hf.close()


if __name__ == '__main__':
    newspapers_dict = get_text_dirs()
    TEXT_DIRS_nahar = newspapers_dict['nahar']
    create_archive_hdf5(TEXT_DIRS_nahar, archive='nahar')


    # newspapers_dict = get_text_dirs()
    # TEXT_DIRS_nahar = newspapers_dict['nahar']
    # TEXT_DIRS_HAYAT = newspapers_dict['hayat']
    # TEXT_DIRS_ASSAFIR = newspapers_dict['assafir']
    #
    # # creating hdf5 for archives
    # # create_archive_hdf5(TEXT_DIRS_nahar, archive='nahar')
    # # create_archive_hdf5(TEXT_DIRS_HAYAT, archive='hayat')
    # create_archive_hdf5(TEXT_DIRS_ASSAFIR, archive='assafir')

    # hf = h5py.File('nahar.h5', 'r')
    # for group in hf.keys():
    #     print('-------------------------- group: {} ------------------------------------'.format(group))
    #     years = []
    #     for dset in hf[group].keys():
    #         # hf['1978']['78010802.txt'].value
    #         # print('dest: {}'.format(dset))
    #         years.append(dset[:2])
    #     print(set(years))
            # print('len: {}'.format(len(hf[group][dset].value)))

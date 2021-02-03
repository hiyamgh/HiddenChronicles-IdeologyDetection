import os
import glob
import h5py
import getpass
import nexusformat.nexus as nx
from text_dirs import get_text_dirs


def create_h5py_data():
    hf = h5py.File('data.h5', 'w')
    MAIN_DIR = 'sample_txt/'
    # get list of all possible years
    for issuepage in os.listdir(MAIN_DIR):
        if issuepage[0].isdigit():
            prefix = issuepage[:6]
            year = issuepage[:2]
            month = issuepage[2:4]
            groupID = None
            if int(year) < 20:
                year_str = '20{}'.format(year)
                if year_str not in hf.keys():
                    groupID = hf.create_group(year_str)
            else:
                year_str = '19{}'.format(year)
                if year_str not in hf.keys():
                    groupID = hf.create_group(year_str)

            text_file = open(os.path.join(MAIN_DIR, issuepage), encoding='utf-8')
            dt = h5py.special_dtype(vlen=str)
            dset = groupID.create_dataset(issuepage[:-4], dtype=dt, data=text_file.read())
            dset.attrs['month'] = month
            dset.attrs['year'] = year_str
    
    hf.close()


def create_archive_hdf5(TEXT_DIRS, archive):
    ''' create text dirs for a certain archive '''
    hf = h5py.File('{}.h5'.format(archive), 'w')
    for txt_dir in TEXT_DIRS:
        count = 0
        print('processing files in {}'.format(txt_dir))
        for issuepage in os.listdir(txt_dir):
            if issuepage[0].isdigit() and issuepage.endswith('.txt'):
                prefix = issuepage[:6]
                year = issuepage[:2]
                month = issuepage[2:4]
                day = issuepage[4:6]
                pagenb = issuepage[6:8]
                groupID = None

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
                dt = h5py.special_dtype(vlen=str)
                dset = groupID.create_dataset(issuepage[:-4], dtype=dt, data=text_file.read())
                dset.attrs['year'] = year_str
                dset.attrs['month'] = month
                dset.attrs['day'] = day
                dset.attrs['pagenb'] = pagenb

                count += 1

                if count%1000 == 0:
                    print('processed {} files so far'.format(count))

    hf.close()


if __name__ == '__main__':
    newspapers_dict = get_text_dirs()
    TEXT_DIRS_nahar = newspapers_dict['nahar']
    TEXT_DIRS_HAYAT = newspapers_dict['hayat']
    TEXT_DIRS_ASSAFIR = newspapers_dict['assafir']

    # creating hdf5 for archives
    # create_archive_hdf5(TEXT_DIRS_nahar, archive='nahar')
    # create_archive_hdf5(TEXT_DIRS_HAYAT, archive='hayat')
    create_archive_hdf5(TEXT_DIRS_ASSAFIR, archive='assafir')


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

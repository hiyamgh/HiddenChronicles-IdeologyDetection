import h5py
import glob
import numpy as np
import os


def nahar_to_HDF5(nahar_batches, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # the HDF5 file
    h5_file = h5py.File(output_folder + 'hayat.hdf5', 'w')

    # combine all batches into 1 list
    nahar_batch1 = glob.glob(nahar_batches[0] + "*.txt")
    nahar_batch2 = glob.glob(nahar_batches[1] + "*.txt")
    nahar_batch3 = glob.glob(nahar_batches[2] + "*.txt")
    nahar_batch4 = glob.glob(nahar_batches[3] + "*.txt")

    nahar = nahar_batch1 + nahar_batch2 + nahar_batch3 + nahar_batch4

    dt = h5py.special_dtype(vlen=str)
    # txt_files = glob.glob(assafir + "*.txt")

    # create dataset of (newspaper_id, newspaper_txt)
    # newspaper_id: the year-month-date-pagenb of the newspaper
    # the string from the txt file of the newspaper
    ds = h5_file.create_dataset('as', (len(nahar),), dtype=np.dtype([("astring1", dt),('astring2', dt)]))

    for idx, file in enumerate(nahar):
        # get the name of the file (year + day + month + page number)
        date_page = file[-12:-4]
        print('newspaper id: %s' % date_page)

        file = open(file, "r+", encoding="utf8")
        # read the text in the file
        text = file.readlines()
        text = ' '.join(text)

        ds[idx] = (date_page, text)

    h5_file.close()

    print('Created file: %s of size: %.3f GB' % ('nahar.hdf5', convert_unit(os.path.getsize('HDF5_files/nahar.hdf5'), 'GB')))


def convert_unit(size_in_bytes, unit):
    """ Convert the size from bytes to other units like KB, MB or GB"""
    if unit == 'KB':
        return size_in_bytes / 1024
    elif unit == 'MB':
        return size_in_bytes / (1024 * 1024)
    elif unit == 'GB':
        return size_in_bytes / (1024 * 1024 * 1024)
    else:
        return size_in_bytes


if __name__ == '__main__':
    batches = ['E:/newspapers/nahar/nahar/nahar-batch-1/out/',
              'E:/newspapers/nahar/nahar/nahar-batch-2/out/',
              'E:/newspapers/nahar/nahar/nahar-batch-3/out/'
              'E:/newspapers/nahar/nahar/nahar-batch-4/out/']

    nahar_to_HDF5(nahar_batches=batches, output_folder='HDF5_files/')
    print('Created file: %s of size: %.3f GB' % ('nahar.hdf5', convert_unit(os.path.getsize('HDF5_files/nahar.hdf5'), 'GB')))




import h5py
import glob
import numpy as np
import os


def assafir_to_HDF5(assafir_batches, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # the HDF5 file
    h5_file = h5py.File(output_folder + 'assafir.hdf5', 'w')

    # combine both batches into 1 list
    assafir_batch1 = glob.glob(assafir_batches[0] + "*.txt")
    assafir_batch2 = glob.glob(assafir_batches[1] + "*.txt")
    assafir = assafir_batch1 + assafir_batch2

    dt = h5py.special_dtype(vlen=str)
    # txt_files = glob.glob(assafir + "*.txt")

    # create dataset of (newspaper_id, newspaper_txt)
    # newspaper_id: the year-month-date-pagenb of the newspaper
    # the string from the txt file of the newspaper
    ds = h5_file.create_dataset('assafir', (len(assafir),), dtype=np.dtype([("astring1", dt),('astring2', dt)]))

    for idx, file in enumerate(assafir):
        # get the name of the file (year + day + month + page number)
        date_page = file[-12:-4]
        print('newspaper id: %s' % date_page)

        file = open(file, "r+", encoding="utf8")
        # read the text in the file
        text = file.readlines()
        text = ' '.join(text)

        ds[idx] = (date_page, text)

    h5_file.close()

    print('Created file: %s of size: %.3f GB' % ('assafir.hdf5', convert_unit(os.path.getsize('HDF5_files/assafir.hdf5'), 'GB')))


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
    batches = ['E:/newspapers/assafir/assafir/assafir-batch-1/out/',
                'E:/newspapers/assafir/assafir/assafir-batch-2/out/']

    # assafir_to_HDF5(assafir_batches=batches, output_folder='HDF5_files/')
    print('Created file: %s of size: %.3f GB' % ('assafir.hdf5', convert_unit(os.path.getsize('HDF5_files/assafir.hdf5'), 'GB')))




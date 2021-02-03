"""
<keywords>
test, python, hdf5, h5py, write, image, compression
</keywords>
<description>
</description>
<seealso>
</seealso>
"""
import os
import glob
import numpy
import h5py

data_path = '/home/mher/tmp/sandbox/home/lh13/test/classification/code/data/input/Datasetdistinct1250CleanedMin16'

import time
t0 = time.time()
data = []
for file_count, fpath in enumerate(glob.iglob(os.path.join(data_path, '**', '*.txt'), recursive=True)):
    print(fpath)
    with open(fpath) as fobj:
        data.append(fobj.read())
t1 = time.time()
print()

with h5py.File('data.h5', 'w') as h5fobj:

    for file_count, fpath in enumerate(glob.iglob(os.path.join(data_path, '**', '*.txt'), recursive=True)):
        print(fpath)
        with open(fpath) as fobj:
            dirname = os.path.split(os.path.dirname(fpath))[-1]
            fname = os.path.split(fpath)[-1]
            key = os.path.join(dirname, fname)

            h5fobj.create_dataset(
                key,
                data=numpy.frombuffer(fobj.read().encode(), dtype=numpy.uint8),
                dtype=numpy.uint8,
                compression='gzip',
                compression_opts=9
            )

t0 = time.time()
data = []
with h5py.File('data.h5') as h5fobj:
    keys = list(h5fobj.keys())
    for key in keys:
        fnames = list(h5fobj[key].keys())
        for fname in fnames:
            fpath = os.path.join(key, fname)
            _data = numpy.array(h5fobj[fpath]).tobytes().decode()
            data.append(_data)
t1 = time.time()

print('done')
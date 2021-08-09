import h5py

hf = h5py.File('sample.h5', 'w')
years = [2019, 2020, 2021]
string_dt = h5py.special_dtype(vlen=str)
for y in years:
    grp = hf.create_group(str(y))
    grp.create_dataset('dataset_{}'.format(y), dtype=string_dt, data='This is year {} hooray'.format(y))
hf.close()

hf = h5py.File('sample.h5', 'r')
for k in hf.keys():
    print(k)

#|%%--%%| <GRnB486iUi|wnrSjahv6C>

# %%


import h5py
from codetiming import Timer
import dill
import os
import matplotlib.pyplot as plt

from modules.ajh_utils import handy_dandy as hd
import numpy as np
from modules.ModelTools import TrainingTools as tt
from modules.ProcessingTools import FitsProcessing as fp
from modules.ajh_utils import lineplots as lplts
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsRegressor
import blosc
import lzma

from modules.ProcessingTools import ByteWriter

# %%


#|%%--%%| <wnrSjahv6C|d64KS9hlrB>
# %%

# with open("./datasets/models/knnOne_training_testing.dill", "rb") as f:
#     knn, filtered_train, filtered_test = dill.load(f)

with open('./datasets/multisets/MultiSet_One.dill', 'rb') as f:
    multiset = dill.load(f)

#|%%--%%| <d64KS9hlrB|JzOvB8YtSo>
# %%
timer = Timer(name='class')
timer.start()


ByteWriter.dump(multiset, './datasets/MultiSets/MultiSet_One.dill.blosc')

timer.stop()
# compressed_multiset = blosc.compress(pickled_multiset, cname='lz4')

#|%%--%%| <JzOvB8YtSo|nVfq1XCW24>
# %%
timer = Timer(name='class')
timer.start()

multiset2 = ByteWriter.load('./datasets/multisets/MultiSet_One.dill.blosc')

timer.stop()

#|%%--%%| <nVfq1XCW24|o0ADpKmQee>
# %%

n_bytes = len(pickled_multiset)
max_bytes = int( np.round( blosc.MAX_BUFFERSIZE / 2 ) )

#|%%--%%| <o0ADpKmQee|AoV7g2uTkP>
# %%
## tests of various dumping and compression methods

## splitting pickle-byte obj into chunks
timer.start()
## write in byte chunks
with open('./datasets/ModelSets/multisetone.bdill', 'wb') as f_out:
    for idx in range(0, len(pickled_multiset), max_bytes):
        f_out.write(pickled_multiset[idx:idx+max_bytes])

timer.stop()
# %%
#|%%--%%|



## write using lzma compression
timer = Timer(name='class')
timer.start()
with lzma.open('./datasets/multisets/multisetone.dill.xz', 'wb') as f_out:
    dill.dump(multiset, f_out, byref=True)

timer.stop()

#|%%--%%|


modelset = tt.ModelSet(knn, multiset, filtered_train, filtered_test)

## write using lzma compression
timer = Timer(name='class')
timer.start()
with lzma.open('./datasets/ModelSets/KNN_One.dill.xz', 'wb') as f_out:
    dill.dump(modelset, f_out, byref=True)

timer.stop()

#|%%--%%|
## checking read  times

timer.start()
with lzma.open('./datasets/ModelSets/KNN_One.dill.xz', 'rb') as f_in:
    modelset_loaded = dill.load(f_in)

timer.stop()


#|%%--%%|

timer.start()
print( dill.dumps(modelset) == dill.dumps(modelset_loaded) )

timer.stop()

#  ----------------------------------------------------------------------------------------
## see Scratch/BigFileCompression.py for an exploration of the different
# object dumping methods I explore
# ----------------------------------------------------------------------------------------

#|%%--%%|


## converting base project class types to hdf5 storable formats

with lzma.open('./datasets/multisets/multisetone.dill.xz', 'rb') as f:
    multiset = dill.load(f)


#|%%--%%|

multiset_dict = multiset.__dict__
sources_dicts = [source.__dict__ for source in multiset_dict['source_filesets']] 


#|%%--%%|
multiset.filename = 'none'

dt = h5py.special_dtype(vlen=str)
with h5py.File('./multisets.h5', 'w') as f:
    m1 = f.create_group('multiset_one')
    for key in multiset_dict:
        if not multiset_dict[key]:
            continue

        if key != 'source_filesets' :
            key_type = type(multiset_dict[key])
            # print(f'{key_type = }')
            # print(f'{key = }')
            m1.create_dataset(key, data=multiset_dict[key], shape=(1,))
        elif key == 'source_filesets':
            src_grp = m1.create_group('source_filesets')

            for fileset in multiset_dict[key]:
                fileset_dict = fileset.__dict__
                filename = hd.getFileName(fileset_dict["source_filename"])
                fileset_grp = src_grp.create_group(filename)
                os.makedirs(f'./datasets/multisets/one/headers/{filename}')

                for fkey in fileset_dict:
                    if fkey != 'training_set' and fkey != 'testing_set' and fkey != 'headers' and fkey != 'fwhm' and fkey != 'fwhms':
                        print(f'{fkey = }')
                        print(f'{type( fkey ) = }')
                        print(f'{type( fileset_dict[fkey]  ) = }')
                        # if type( fileset_dict[fkey] ) == type(list()):
                        #     fileset_dict[fkey] = np.array(fileset_dict[fkey])
                        #    print( fileset_dict[fkey].shape )
                        if type(fileset_dict[fkey]) == type(float()):
                            fileset_grp.create_dataset(fkey, data=[ fileset_dict[fkey] ], dtype='f')
                        elif type(fileset_dict[fkey]) == type(str()):
                            fileset_grp.create_dataset(fkey, data=[ fileset_dict[fkey] ], dtype=dt)
                        elif type(fileset_dict[fkey]) == type(np.array([])):
                            fileset_grp.create_dataset(fkey, data=[ fileset_dict[fkey] ])

                    # elif  fkey == 'headers':
                    #     headers = fileset_dict[fkey]
                    #     for header in headers:
                    #         # print(header)
                    #         coords = header.input_position_original
                    #         os.mkdir(f'./datasets/multisets/one/headers/{filename}/{coords}')
                    #         with open(f'./datasets/multisets/one/headers/{filename}/{coords}/cutout.dill', 'wb') as f_out:
                    #             dill.dump(header, f_out)



    # [print(name ) for name in m1['source_filesets']]

    # print(m1['source_filesets'][:]['date_created'])
    

#|%%--%%|

with h5py.File('./multisets.h5', 'r') as f:
    m1 = f['multiset_one']
    for fname in m1['source_filesets']:
        fileset = m1['source_filesets'][fname]
        # print(fileset)
        for fattr in fileset:
            # print(fattr)

            print(fileset[fattr][0])
        



#|%%--%%|


#|%%--%%|

with h5py.File('./test.hdf5', 'r') as f:
    dset = f['header']
    # print(dset[0])

    h1_bstr_loaded = str( dset[0] )
    print(blosc.compress(dill.dumps(h1)))
    print(f'org = {h1_bstr}')
    print(f'new = {h1_bstr_loaded}')
    print(f'new = {h1_bstr_loaded[2:len(h1_bstr_loaded) - 1]}')






#|%%--%%|
cutout = multiset.source_filesets[0].headers[0]
# pkl = dill.dumps(cutout)

import codecs
pickled = codecs.encode(dill.dumps(cutout), "base64").decode()

unpickle = lambda pickled : dill.loads(codecs.decode(pickled.encode(), "base64"))

with h5py.File('dilled.h5', 'w') as f:
    f.create_dataset('cutout', data=[ pickled ])

with h5py.File('dilled.h5', 'r') as f:
    dset = f['cutout']
    # dset[0].replace('b', '')
    print(dill.dumps( unpickle(dset[0].decode())) == dill.dumps(cutout))

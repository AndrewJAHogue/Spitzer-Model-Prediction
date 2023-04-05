
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



#|%%--%%|
    

#|%%--%%|
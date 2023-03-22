#|%%--%%|


import lzma
import h5py
import bloscpack as bp
from codetiming import Timer
import joblib as jbl
import dill
import pickle
import numpy as np
# import blosc
import blosc2 as blosc

class test:
    def __init__(self, data1, data2) -> None:
         self.first = data1
         self.sec = data2


#|%%--%%|


# ---------------------------------------------------------------------------- 
## writing to blosc compressed object in byte chunks
timer = Timer(name='class')
timer.start()

# random placeholder data
data = np.random.rand(3, 3, int( 1e7 ))
pickled_data = dill.dumps(data, byref=True)

n_bytes = len(pickled_data)
max_bytes = int( np.round( blosc.MAX_BUFFERSIZE / 2 ) )

bytes_array = bytearray()

## write
for idx in range(0, len(pickled_data), max_bytes):
    bytes_array.extend( blosc.compress(pickled_data[idx:idx+max_bytes]) )

bytes_string = bytes(bytes_array)

with open('./datasets/ModelSets/test.dill.blosc', 'wb') as f_out:
    f_out.write(bytes_string)

timer.stop()

# %%
## read
import os
timer = Timer(name = 'class')
timer.start()

filepath = './datasets/ModelSets/test.dill.blosc'
bytes_in = bytearray(0)
input_size = os.path.getsize(filepath)
with open(filepath, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)

pickled_data = blosc.decompress(bytes_in)

data2 = dill.loads(pickled_data)

timer.stop()

# %%
## testing regular compression
timer = Timer(name='class')
timer.start()
with open(filepath, 'wb') as f_out:
    dill.dump(data, f_out)

timer.stop()

#|%%--%%|

data = lambda : np.random.rand(5, 5, int( 1e7 ))
data2 = test(test(data(), data()), data())

#|%%--%%|
timer = Timer(name='class')
timer.start()

pickled = dill.dumps(data2)
# pickled = pickle.dumps(data2)
print(f'{len(pickled) = }')
compressed = blosc.compress2(pickled)

timer.stop()

#|%%--%%|
timer = Timer(name='class')
timer.start()
decompressed = blosc.decompress(compressed)
print(f'{len(decompressed) = }')
unpickled = dill.loads(decompressed)


timer.stop()
## decompress2 and dill return "UnpicklingError: pickle data was truncated"

#|%%--%%|

## lets try using regular decompress and splitting it into multiple files to compress and save

pickled = dill.dumps(data2)
print(f'{len(pickled) = }')

#|%%--%%|



# timer = Timer(name='class')
# timer.start()
# if len(pickled) >= blosc.MAX_BUFFERSIZE:
file_bytes = len(pickled)
n_chunks = int( np.ceil(file_bytes / blosc.MAX_BUFFERSIZE) )
increment = int( np.ceil( file_bytes / n_chunks ) )

#|%%--%%|


i = 0
for pickled_byte in range(0, file_bytes, increment):
    chunk = pickled[pickled_byte : pickled_byte + increment]
    compressed = blosc.compress(chunk)
    with open(f'./test{i}.dill.blosc', 'wb')  as f_out:
        f_out.write(compressed)
    
    i += 1

# timer.stop()

#|%%--%%|
## not working
bytes_arr = bytearray(0)

for n in range(0,4):
    print(f'{n = }')
    with open(f'./test{n}.dill.blosc', 'rb') as f_in:
        chunk = blosc.decompress(f_in.read())
        bytes_arr += chunk

    bytes_str = bytes(bytes_arr)

    data_loaded = dill.loads(bytes_str)

#|%%--%%|
# trying bloscpack
timer = Timer(name='class')
timer.start()

pickled = dill.dumps(data2)
print(f'{len(pickled) = }')

bp.pack_bytes_to_file(pickled, './test.dill.blp')

timer.stop()

#|%%--%%|
## testing out HDF5


#|%%--%%|
# creating the file
f = h5py.File('./test.hdf5', 'w')
dset = f.create_dataset('testdataset', (100))
#|%%--%%|
# appending the file
f = h5py.File('./test.hdf5', 'a')
grp = f.create_group('subgroup')

#|%%--%%|


# can't keep complex datatypes
dset2 = grp.create_dataset('testfunc', dtype='')

#|%%--%%|


dset2[0] = data2



#
dset 


# f.close()

#|%%--%%|

# ----------------------------------------------------------------------------------------
#|%%--%%|

#|%%--%%|


import dill
import numpy as np
import blosc
from codetiming import Timer
import lzma

def dump(data, file_out):
    timer = Timer(name='class')
    timer.start()
    pickled_data = dill.dumps(data, byref=True)

    n_bytes = len(pickled_data)
    max_bytes = int( np.round( blosc.MAX_BUFFERSIZE / 2 ) )

    bytes_array = bytearray()

    ## write

    # with compression
    # for idx in range(0, n_bytes, max_bytes):
    #     bytes_array.extend( blosc.compress(pickled_data[idx:idx+max_bytes]) )

    # without compression
    for idx in range(0, n_bytes, max_bytes):
        bytes_array.extend(pickled_data[idx:idx+max_bytes] )

    bytes_string = bytes(bytes_array)

    with open(file_out, 'wb') as f_out:
        f_out.write(bytes_string)

    timer.stop()


def load(file_in):
    timer = Timer(name='class')
    import os
    timer.start()

    max_bytes = int( np.round( blosc.MAX_BUFFERSIZE / 2 ) )
    
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_in)
    with open(file_in, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            # bytes_in += blosc.decompress( f_in.read(max_bytes) )
            bytes_in += f_in.read(max_bytes)

    # pickled_data = blosc.decompress(bytes( bytes_in ))
    # pickled_data = bytes(bytes_in)
    timer.stop()
    return dill.loads(bytes(bytes_in))
    # return dill.loads(pickled_data)


class test:
    def __init__(self, data1, data2) -> None:
         self.first = data1
         self.sec = data2

#|%%--%%| <DOmteydIvc|SApauvYCDb>
if __name__ == '__main__':

    test1 = test( np.random.rand(3,3, int(1e7)),  np.random.rand(3,3, int(1e7)) )
    # filepath = './datasets/multisets/test.dill.blosc'
    filepath = '../../datasets/multisets/test.dill.blosc'

    dump(test1, filepath)

    #|%%--%%| <SApauvYCDb|bLPA3BXXYX>

    test_loaded = load(filepath)

    #|%%--%%| <bLPA3BXXYX|WRAtENp59x>
    timer = Timer(name='class')
    timer.start()

    test1 = test( np.random.rand(5,5, int(1e7)),  np.random.rand(5,5, int(1e7)) )
    test2 = test(test1, np.random.rand(5,5, int(1e7)))


    filepath = '../../datasets/multisets/test.dill.lzma'
    with lzma.open(filepath, 'wb') as f_out:
        dill.dump(test2, f_out)


    timer.stop()
    #|%%--%%| <WRAtENp59x|y4HGwZv9vN>

    test_loaded = load(filepath)

    #|%%--%%| <y4HGwZv9vN|khIt1p54QA>
    ## this is not as accurate as comparing bytes
    print( np.all([test1.first, test_loaded.first]) )
    print( np.all([test1.sec, test_loaded.sec]) )
    #|%%--%%| <khIt1p54QA|CoxYSPdfeP>
    ## slowest, but most accurate
    print( dill.dumps(test1) == dill.dumps(test_loaded) )
    #|%%--%%| <CoxYSPdfeP|RnZU3Uli1C>

    with open('./datasets/multisets/MultiSet_One.dill', 'rb') as f_in:
        multiset1 = dill.load(f_in)
    #|%%--%%| <RnZU3Uli1C|0mF3EZMaUb>

    multiset_loaded = load('./datasets/multisets/MultiSet_One.dill.blosc')

    #|%%--%%| <0mF3EZMaUb|tdDHGCGjqV>
    all_train_loaded = multiset_loaded.getTrainingData()
    for i, train in enumerate(multiset1.getTrainingData()):
        if not np.all([train, all_train_loaded[i]]):
            print(f'{i = }')

    #|%%--%%| <tdDHGCGjqV|47IZZe5BEF>

    print( multiset_loaded == multiset1 )

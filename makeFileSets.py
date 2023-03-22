def makeFileSet(file_path, filename):
    from modules.ModelTools.TrainingTools import CreateFileSet

    return CreateFileSet(file_path, filename, peak_percentage=0.5, sigma=0., fwhm=10., threshold=10.)

def my_func(file_path, filename):
    from modules.ProcessingTools.FitsProcessing import getFWHMsForFile
    from astropy.io import fits
    import numpy as np

    try:
        dataset = makeFileSet(file_path, filename)
        fits_file = fits.open(file_path)

    except:
        print(f'--------------------------------------------{filename} failed--------------------------------------------')
        return  np.NaN, np.NaN

    return getFWHMsForFile(dataset, fits_file)


def makeFileSets(base_dir):
    from multiprocess import pool
    # from multiprocessing import  get_context, pool, cpu_count
    from codetiming import Timer
    import os
    import dill

    all_files = os.listdir(base_dir)

    timer = Timer(name="class")
    timer.start()
    
    results = []
    fwhms = []
    # with get_context('spawn').Pool(processes=6) as p:
    with pool.Pool(processes=8) as p:
        print('starting next iteration')
        results = [ p.apply_async(my_func, args=(base_dir+f, f,)) for f in all_files ]

        for r in results:
            try:
                fwhms, dataset = r.get()
                dataset.fwhm = fwhms

                # print(f'{fwhms = }')
                # dataset.saveFileSet()
                dataset.UpdateFileSetTime()
                with open(f'./datasets/cutouts/dills/{dataset.source_filename}_training_testing_headers.dill', 'wb') as f:
                    dill.dump(dataset, f, byref=True)
            except:
                print('Result Failed')

            

    timer.stop()
#|%%--%%| <pLFsC6jZUX|4uNdGi2qh4>


# %%

# if __name__ == '__main__':
# makeFileSets('../../IRSA_Spitzer_Mipsgal/datasets/')

import dill
import os

from modules.ajh_utils import handy_dandy as hd
import numpy as np
from modules.ModelTools import TrainingTools as tt
from modules.ProcessingTools import FitsProcessing as fp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsRegressor

# %%


with open('./datasets/multisets/MultiSet_One.dill', 'rb') as f:
    multiset = dill.load(f)

# %%
#|%%--%%| <4uNdGi2qh4|GsJMvcTBvQ>

training = multiset.getTrainingData()
testing = multiset.getTestingData()
# filtered_train = np.concatenate(filtered_train)
# filtered_test = np.concatenate(filtered_test)
    


filtered_train, filtered_test = fp.Filter(np.concatenate(training), np.concatenate(testing), std_coefficient=1, sigma=0)
filtered_train = fp.SimpleProcessData(filtered_train, 0)  
filtered_test = fp.SimpleImpute(filtered_test)  

#|%%--%%| <GsJMvcTBvQ|14Gz6ZDSIp>


if len(filtered_train):

    # filtered_train = np.array(filtered_train)
    # filtered_test = np.array(filtered_test)
    
    x_train, x_test, y_train, y_test = train_test_split(
        filtered_train, filtered_test, test_size=0.2
    )


    knn = KNeighborsRegressor()
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print(f'{score = }')





    import matplotlib.pyplot as plt
    from modules.ajh_utils import lineplots as lplts

    pred = knn.predict(x_test)
    lplts.plot_gallery(pred, 50, 50, 30, 4, stats=True)
    # plt.show()

# %%

with open('./datasets/models/knnOne_training_testing.dill', 'wb') as f:
    dill.dump((knn, filtered_train, filtered_test), f)
    
    
# %%

%load_ext autoreload
%reload_ext autoreload
%autoreload 2

    
zipped = np.array( hd.myzip(x_test, y_test, pred) )
# %%

lplts.plot_gallery(zipped, 50, 50, 100, 3, index=False, stats=True)


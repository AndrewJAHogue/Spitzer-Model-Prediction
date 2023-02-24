''' Focusing on processing the MIPSGAL fits file 'MG0000n005_024.fits' '''


# %%
#|%%--%%| <JxJodmThWa|uxIZmlyuQa>


%load_ext autoreload
%reload_ext autoreload
%autoreload 2
import os
from astropy.io import fits
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from modules.ajh_utils import lineplots as lplts
import matplotlib.pyplot as plt
import numpy as np
import contextlib
from modules.ModelTools import TrainingTools as tt
from astropy.io import fits
import matplotlib.pyplot as plt
from modules.ProcessingTools import FitsProcessing as fp

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import  KNeighborsRegressor
from modules.ajh_utils import handy_dandy as hd
from modules.ajh_utils import lineplots as lplts
import numpy as np
from astropy.stats import sigma_clipped_stats

FILE_DIR = './datasets/MG/'
FILENAME = 'MG0000n005_024.fits'
FILENAME_1 = 'MG0000p005_024.fits'
FILENAME_2 = 'MG0000p015_024.fits'

# %%


sigma = 20
nsigma = 10
radius = 1
fwhm = 10
threshold = 10


file = fits.open(f'{FILE_DIR}{FILENAME}')
file_data = file[0].data
file1 = fits.open(f'{FILE_DIR}{FILENAME_1}')
file_data1 = file1[0].data
file2 = fits.open(f'{FILE_DIR}{FILENAME_2}')
file_data2 = file2[0].data
# cutouts, headers = hd.createCutoutsList(file_data)

#|%%--%%| <uxIZmlyuQa|fYnYY8pxVI>

plt.imshow(file_data2)
#|%%--%%| <fYnYY8pxVI|p6A8wv1PuO>




## move on to masking the sources

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, detect_threshold
from photutils.utils import circular_footprint



# %%

# ## model training on this 

# def processAndMask(file, **keywargs):
#     """_summary_

#     Args:
#         file (string) : string file path to joblib file of (cutouts, headers)
#         sigma (int) : 
#         radius (int) :
#         nsigma (int) :


#     Returns:
#         tuple[3] : (string file path, processed_data, masked_data)
#     """    

#     sigma = keywargs.get('sigma', 3)
#     radius  = keywargs.get('radius', 10)
#     nsigma = keywargs.get('nsigma', 10)

#     import time

#     time.sleep(1)
#     try:
#         return process_and_mask(file, sigma=sigma, nsigma=nsigma, radius=radius)
#     except:
#         print(f'{file} failed')


# def process_and_mask(file, **keywargs):
#     """_summary_

#     Args:
#         file (string) : string file path to joblib file of (cutouts, headers)
#         sigma (int) : 
#         radius (int) :
#         nsigma (int) :


#     Returns:
#         tuple[3] : (string file path, processed_data, masked_data)
#     """    

#     from modules.ajh_utils import handy_dandy as hd

#     sigma = keywargs.get('sigma', 3)
#     radius  = keywargs.get('radius', 10)
#     nsigma = keywargs.get('nsigma', 10)

#     print(f'{file} loaded')
#     if '.jbl' in file or '.joblib' in file:
#         data, headers = joblib.load(f'{file}')
#     elif '.fits' in file:
#         file_data = fits.getdata(file)
#         data, headers = hd.createCutoutsList(file_data)
#     else:
#         raise AttributeError(f'{ file } is not an accepted file extension type')
#         
#     

#     processed_data = []
#     masked_data = []
#     processed_data = hd.processData(data.reshape(-1,1) )
#     processed_data = processed_data.reshape(-1, 50, 50)

#     masked_data = hd.mask_sources(processed_data, sigma=sigma, nsigma=nsigma, radius=radius)


#     return (file, processed_data , masked_data)

# # %%

# def trainModel(testing_data, training_data):
#     # reshape the data
#     testing_data = testing_data.reshape(-1, 2500)
#     training_data = training_data.reshape(-1, 2500)

#     # split data up
#     input_train, input_test, output_train, output_test = train_test_split(training_data, testing_data, test_size=0.2, shuffle=False)

#     # use the RidgeCV model
#     rcv = RidgeCV()
#     rcv.fit(input_train, output_train)

#     return rcv






# # %%

# sigma = 3.
# nsigma = 10.
# fwhm = 10.
# threshold = 5.
# radius = 1

# %%
#|%%--%%| <p6A8wv1PuO|xMT9ARgukk>

from astropy.wcs import WCS
from astropy import units as u


chosen_file = file1[0]

wcs = WCS(chosen_file.header)
# pt2 = wcs.pixel_to_world(3168, 3168).galactic

# %matplotlib inline 
ax = plt.subplot(121, projection = wcs)
plt.imshow(chosen_file.data)

overlay = ax.get_coords_overlay('galactic')
overlay.grid(color='white', ls='dotted')

plot2 = plt.subplot(122)
plot2.imshow(chosen_file.data)
plot2.invert_yaxis()

plt.show()

# %%
#|%%--%%| <xMT9ARgukk|t0wSTX6pex>





#|%%--%%| <t0wSTX6pex|db2rcjPs16>
# %%


#|%%--%%| <db2rcjPs16|qv9oFmsBmv>
# %%

#|%%--%%| <qv9oFmsBmv|iZicQTMuTW>
# %%

file = fits.open(f'{FILE_DIR}{FILENAME}')
file_data = file[0].data
file1 = fits.open(f'{FILE_DIR}{FILENAME_1}')
file_data1 = file1[0].data
file2 = fits.open(f'{FILE_DIR}{FILENAME_2}')
file_data2 = file2[0].data

## --------------------FILE SPECIFIC MASK--------------------------------------------------------------------------------------------------------------------------
file_data = fp.sliceImageProperly(file)
file_data1 = fp.sliceImageProperly(file1)
file_data2 = fp.sliceImageProperly(file2)
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------




sigma = 0.
# nsigma = 1000.
fwhm = 10.
threshold = 10.
radius = 1
set1 = tt.CreateFileSet(file_data2, FILENAME_2, peak_percentage=0.5, sigma=sigma,fwhm=fwhm, threshold=threshold)
training, testing = set1.getData()
print(f'{training.shape = }')

filtered_training, filtered_testing = fp.Filter(training, testing, std_coefficient=1, sigma=0)
print(f'{filtered_training.shape = }')

# set1.saveFileSet()

# %%

from codetiming import Timer

timer = Timer(name="class")
timer.start()
fwhms = []
for i,t in enumerate( testing ):
    try:
        f = hd.get_fwhm(t)
    except ValueError:
        f = np.NaN
        
    pos = set1.headers[i].input_position_original
    fwhms.append((pos, f))

timer.stop()


#|%%--%%| <iZicQTMuTW|07mdRkIJPs>

from codetiming import Timer
timer = Timer(name="class")
timer.start()

fwhms = []
# for i,t in enumerate( testing ):
def fwhm(t, i):
    try:
        f = hd.get_fwhm(t)
    except ValueError:
        f = np.NaN
        
    pos = set1.headers[i].input_position_original
    fwhms.append((pos, f))

from multiprocess import pool

with pool.Pool(processes=4) as p:
    results = [p.apply_async(fwhm, args=(t,i,)) for i,t in enumerate(testing)]

    print(results[0].get())
timer.stop()


# %%



lplts.plot_gallery(filtered_training, 50, 50, 5, 5, stats=True)


# %%
#|%%--%%| <07mdRkIJPs|WTXcoIHOl2>

filtered_training = fp.SimpleProcessData(filtered_training, sigma)
filtered_testing = fp.SimpleImpute(filtered_testing)

# lplts.plot_gallery(filtered_training, 50, 50, 50, 6, stats=True)

print(f'{training.shape = }')
print(f'{filtered_training.shape = }')

# %%



x_train, x_test, y_train, y_test = train_test_split(
    filtered_training, filtered_testing, test_size=0.2
)

# rcv = RidgeCV()
# rcv.fit(x_train, y_train)
# score = rcv.score(x_test, y_test)
# print(f'{score = }')

## using knn

knn = KNeighborsRegressor()
knn.fit(x_train, y_train)
score = knn.score(x_test, y_test)
print(f'{score = }')

# %%



pred = knn.predict(x_test)
lplts.plot_gallery(pred, 50, 50, 50, 4, stats=True)
plt.show()



#|%%--%%| <a1iNC0Ont6|SD5axJzRIo>




def Stats(input_data, sigma):
    """
    return mean, median, std
    """
    from numpy import nanstd
    from astropy.stats import sigma_clipped_stats


    return sigma_clipped_stats(input_data, sigma=sigma, stdfunc=nanstd)




# %%

from astropy.stats import sigma_clipped_stats

print(sigma_clipped_stats( file_data, stdfunc=np.nanstd ))
# %%

std_coefficient = 1
sigma = 0
filterfunc = lambda a : fp.FirstDerivative(a)
# filterfunc = lambda a : a

from astropy.stats import sigma_clipped_stats
import contextlib


# filter out bad cutouts
if sigma == 0:
    mean, med, std = sigma_clipped_stats(filterfunc(training), stdfunc=np.nanstd)
else:
    mean, med, std = sigma_clipped_stats(filterfunc(training), sigma=sigma, stdfunc=np.nanstd)

print(f'{mean = }')


filtered_training = []
filtered_testing = []
for i, c in enumerate(training):
    c_diff = (filterfunc(c))
    c_mean = np.nanmean(c_diff - med)
    print(f'{i = }')
    print(f'{c_mean = }')
    if c_mean >= (mean - ( std * std_coefficient)):
        with contextlib.suppress(IndexError):
            filtered_training.append(c)
            filtered_testing.append(testing[i])

    if i > 4:
        break
## turn them from lists to np arrays
filtered_training = np.array(filtered_training)
filtered_training = filtered_training.reshape(-1, 2500)

filtered_testing = np.array(filtered_testing)
filtered_testing = filtered_testing.reshape(-1, 2500)
lplts.plot_gallery(filtered_training, 50, 50, 1, 6, stats=True)

# %%

filtered = fp.Filter(training, testing, std_coefficient=10e10, sigma=0, derivfilterfunc=fp.FirstDerivative)[0]
print(f'{training.shape = }')
print(f'{filtered.shape = }')
lplts.plot_gallery(filtered, 50, 50, 1, 6, stats=True)

# %%






# %%

# def Train(fits_file):
#     from sklearn.model_selection import train_test_split
#     from modules.ProcessingTools import FitsProcessing as fp
#     from sklearn.neighbors import  KNeighborsRegressor

#     file_data = fits_file[0].data
#     # slice it up
#     file_data = fp.sliceImageProperly(fits_file)


#     sigma = 0.
#     # nsigma = 1000.
#     fwhm = 10.
#     threshold = 10.
#     radius = 1
#     set1 = tt.CreateFileSet(file_data, peak_percentage=0.5, sigma=sigma,fwhm=fwhm, threshold=threshold)
#     training, testing = set1.getData()
#     print(f'{training.shape = }')

#     filtered_training, filtered_testing = fp.Filter(training, testing, std_coefficient=1, sigma=0)
#     print(f'{filtered_training.shape = }')
        
#     filtered_training = fp.SimpleProcessData(filtered_training, sigma)
#     filtered_testing = fp.SimpleImpute(filtered_testing)

#     x_train, x_test, y_train, y_test = train_test_split(
#         filtered_training, filtered_testing, test_size=0.2
#     )



#     knn = KNeighborsRegressor()
#     knn.fit(x_train, y_train)
#     score = knn.score(x_test, y_test)
#     print(f'{score = }')



#     pred = knn.predict(x_test)
#     lplts.plot_gallery(pred, 50, 50, 50, 4, stats=True)
#     plt.show()

# import os

# files = os.listdir(FILE_DIR)
# print(files)

# for file in files:
#     filepath = FILE_DIR + file

#     Train(fits.open(filepath))

# %%

import dill
import joblib

all_jbls = os.listdir('./datasets/cutouts/jbls')

for f in all_jbls:
    dataset = joblib.load(f'./datasets/cutouts/jbls/{f}')
    with open(f'./datasets/cutouts/dills/{f}.dill', 'wb') as d:
        dill.dump(dataset, d)

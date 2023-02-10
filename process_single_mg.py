''' Focusing on processing the MIPSGAL fits file 'MG0000n005_024.fits' '''


# %%
#|%%--%%| <JxJodmThWa|uxIZmlyuQa>

%load_ext autoreload
%reload_ext autoreload
%autoreload 2
from astropy.io import fits
from astropy.nddata import Cutout2D
import joblib
import matplotlib
from matplotlib import testing
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from modules import ajh_utils 
# from modules import ajh_utils as util
from modules.ajh_utils import lineplots as lplts
from modules.ajh_utils import handy_dandy as util
import matplotlib.pyplot as plt
import numpy as np

FILE_DIR = './datasets/MG/'
FILENAME = 'MG0000n005_024.fits'

# %%


sigma = 20
nsigma = 10
radius = 1
fwhm = 10
threshold = 10


file = fits.open(f'{FILE_DIR}{FILENAME}')
file_data = file[0].data
# cutouts, headers = util.createCutoutsList(file_data)



## move on to masking the sources

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, detect_threshold
from photutils.utils import circular_footprint



# %%

## model training on this 

def processAndMask(file, **keywargs):
    """_summary_

    Args:
        file (string) : string file path to joblib file of (cutouts, headers)
        sigma (int) : 
        radius (int) :
        nsigma (int) :


    Returns:
        tuple[3] : (string file path, processed_data, masked_data)
    """    

    sigma = keywargs.get('sigma', 3)
    radius  = keywargs.get('radius', 10)
    nsigma = keywargs.get('nsigma', 10)

    import time

    time.sleep(1)
    try:
        return process_and_mask(file, sigma=sigma, nsigma=nsigma, radius=radius)
    except:
        print(f'{file} failed')


def process_and_mask(file, **keywargs):
    """_summary_

    Args:
        file (string) : string file path to joblib file of (cutouts, headers)
        sigma (int) : 
        radius (int) :
        nsigma (int) :


    Returns:
        tuple[3] : (string file path, processed_data, masked_data)
    """    

    from modules.ajh_utils import handy_dandy as hd

    sigma = keywargs.get('sigma', 3)
    radius  = keywargs.get('radius', 10)
    nsigma = keywargs.get('nsigma', 10)

    print(f'{file} loaded')
    if '.jbl' in file or '.joblib' in file:
        data, headers = joblib.load(f'{file}')
    elif '.fits' in file:
        file_data = fits.getdata(file)
        data, headers = hd.createCutoutsList(file_data)
    else:
        raise AttributeError(f'{ file } is not an accepted file extension type')
        
    

    processed_data = []
    masked_data = []
    processed_data = util.processData(data.reshape(-1,1) )
    processed_data = processed_data.reshape(-1, 50, 50)

    masked_data = hd.mask_sources(processed_data, sigma=sigma, nsigma=nsigma, radius=radius)


    return (file, processed_data , masked_data)

# %%

def trainModel(testing_data, training_data):
    # reshape the data
    testing_data = testing_data.reshape(-1, 2500)
    training_data = training_data.reshape(-1, 2500)

    # split data up
    input_train, input_test, output_train, output_test = train_test_split(training_data, testing_data, test_size=0.2, shuffle=False)

    # use the RidgeCV model
    rcv = RidgeCV()
    rcv.fit(input_train, output_train)

    return rcv






# %%
import contextlib
from modules.ModelTools import TrainingTools as tt
from astropy.io import fits
import matplotlib.pyplot as plt

sigma = 3.
nsigma = 10.
fwhm = 10.
threshold = 5.
radius = 1

# %%

from astropy.wcs import WCS


wcs = WCS(file[0].header)
# pt2 = wcs.pixel_to_world(3168, 3168).galactic

%matplotlib widget
ax = plt.subplot(121, projection = wcs)
plt.imshow(file_data)

overlay = ax.get_coords_overlay('galactic')
overlay.grid(color='white', ls='dotted')

plot2 = plt.subplot(122)
plot2.imshow(file_data)
plot2.invert_yaxis()

plt.show()

# %%

def getArcMin(ypixel, header):
    from astropy.wcs import WCS
    from astropy import units as u

    wcs = WCS(header)

    # xpixel coord does not matter here, as we only care about the y direction in this case
    return wcs.pixel_to_world(0, ypixel).galactic.b.to(u.arcmin)


def isNearGalCenter(ypixel, header):
    arcmin = getArcMin(ypixel, header)


    return arcmin.value > -20
 
    
def getYMax(fits_file):
    """ A method to find the max ypixel value, the closest we can get to the 
    galactic center, without simply iterating over a multithousand range

    Args:
        fits_file (HDUList): the base fits.open object return

    Returns:
        int: The max of the fits image you can get before crossing our arbitrary 
        "it's now the GC" line
    """    
    file_data = fits_file[0].data
    ymax = file_data.shape[1] 
    yhalf = ymax // 2
    ystep = yhalf // 2

    for y in range(yhalf, ystep + yhalf, 50):
        if isNearGalCenter(y, fits_file[0].header):
            return y

    



# %%
## --------------------FILE SPECIFIC MASK--------------------------------------------------------------------------------------------------------------------------
# file_data = file_data[:2500]
file_data = file_data[:getYMax(file)]
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------


set1 = tt.CreateFileSet(file_data, peak_percentage=0.5)

# set1 = tt.CreateFileSet(f'{FILE_DIR}{FILENAME}')



# %%


from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import  KNeighborsRegressor
from modules.ajh_utils import handy_dandy as util
from modules.ajh_utils import lineplots as lplts
import numpy as np
from astropy.stats import sigma_clipped_stats

# pull data out
training, testing = set1.getData()

lplts.plot_gallery(training, 50, 50, 10, 3)

# %%
## differentiate a cutout
def FirstDerivative(input_data):
    """ Return the first derivative of the 2darray. Appends np.nan after calculation to maintain (50,50) shape

    Args:
        input_data (2d ndarray): a (50,50) cutout

    Returns:
        _type_: _description_
    """    
    return np.diff(input_data, n=1, append=np.nan)


## filter the masked training set based on the standard deviation, mean, and median of the whole training set

def Filter(input_data, std_coefficient=1.5, sigma=3.):
    """ Filter out cutouts that exceed the mean of the whole training set

    Args:
        input_data (tuple of two ndarrays): (input_training_set, input_testing_set) is a tuple of the training and testing data sets, which are both ndarrays containing (50,50) cutouts
        std_coefficient (float, optional): The coefficient to multiply the standard deviation of the whole training set by. Defaults to 1.5.
        sigma (float, optional): The sigma value used for sigma_clipped_stats. Defaults to 3..

    Returns:
        tuple of two ndarrays: A tuple of (training_set, testing_set) that have been filtered
    """    
    training = input_data[0]
    testing = input_data[1]

    # std_coefficient = 1.5 ## coefficient to mult standard deviation by

    # filter out bad cutouts
    mean, med, std = sigma_clipped_stats(input_data, sigma=sigma, stdfunc=np.nanstd)
    filtered_training = []
    filtered_testing = []
    for i, c in enumerate(training):
        c_mean = np.nanmean(c - med)
        print('c_mean condition')
        if c_mean < (mean - ( std * std_coefficient)):
            with contextlib.suppress(IndexError):
                filtered_training.append(c)
                filtered_testing.append(testing[i])

    ## turn them from lists to np arrays
    filtered_training = np.array(filtered_training)
    filtered_training = filtered_training.reshape(-1, 2500)

    filtered_testing = np.array(filtered_testing)
    filtered_testing = filtered_testing.reshape(-1, 2500)

    return filtered_training, filtered_testing


filtered_training, filtered_testing = Filter((training, testing), 1.5, sigma)

lplts.plot_gallery(filtered_training, 50, 50, 5, 6, stats=True)

## alternate imputation method
# filtered_training[np.isnan(filtered_training)] = -1
# filtered_testing[np.isnan(filtered_testing)] = -1
# %%


def SimpleImpute(input_data):
    from numpy import isnan

    input_data[isnan(input_data)] = -1
    
def SimpleProcessData(input_data, sigma):
    from modules.ajh_utils import handy_dandy as hd
    SimpleImpute(input_data)
    return hd.maskBackground(input_data, (50,50), sigma)


filtered_training = SimpleProcessData(filtered_training, sigma)

if len(filtered_training) > 0:
    lplts.plot_gallery(filtered_training, 50, 50, 5, 6, stats=True)

print(f'{training.shape = }')
print(f'{filtered_training.shape = }')


# %%

x_train, x_test, y_train, y_test = train_test_split(
    filtered_training, filtered_testing, test_size=0.3
)

rcv = RidgeCV()
rcv.fit(x_train, y_train)
score = rcv.score(x_test, y_test)
print(f'{score = }')

# %%


pred = rcv.predict(x_test)
%matplotlib inline
lplts.plot_gallery(pred, 50, 50, 50, 4, stats=True)
''' Focusing on processing the MIPSGAL fits file 'MG0000n005_024.fits' '''


# %%

from astropy.io import fits
from astropy.nddata import Cutout2D
import joblib
import matplotlib
from matplotlib import testing
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from modules import ajh_utils 
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

file_data = fits.getdata(f'{FILE_DIR}{FILENAME}')
cutouts, headers = util.createCutoutsList(file_data)



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
## temp cell
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.detection import DAOStarFinder
def getSources(sigma): 

    data = fits.getdata(f'{FILE_DIR}/{FILENAME}')
    mean, med, std  = sigma_clipped_stats(data, sigma=sigma)

    dao = DAOStarFinder(fwhm = 5., threshold=5. * std)
    sources = dao(data - med)

    for col in sources.colnames:
        sources[col].info.format = '%.8g'

    return sources






# %%
from modules.ModelTools import TrainingTools as tt
sigma = 3.
nsigma = 10.
fwhm = 10.
threshold = 5.
radius = 1

FILE_DIR = './datasets/MG/'
FILENAME = 'MG0000n005_024.fits'
set1 = tt.CreateFileSet(f'{FILE_DIR}{FILENAME}')

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from modules.ajh_utils import handy_dandy as util
from modules.ajh_utils import lineplots as lplts
import numpy as np
from astropy.stats import sigma_clipped_stats

# pull data out
training, testing = set1.getData()
# training = util.processData(training, sigma)
# testing = util.processData(testing, sigma)

# filter out bad cutouts
stats = sigma_clipped_stats(testing, sigma=sigma, stdfunc=np.nanstd)
copy = np.copy(training)
print(f'{training.shape = }')
for i, c in enumerate(training):
    mean = np.nanmean(c)
    if mean > (stats[0] + stats[2]):
        try:
            copy = np.delete(copy, i, axis=0)
        except IndexError:
            pass

copy = copy.reshape(-1, 2500)
print(f'{copy.shape = }')


# %%

x_train, x_test, y_train, y_test = train_test_split(
    training, testing, test_size=0.3
)


# rcv = RidgeCV()
# rcv.fit(x_train, y_train)
# score = rcv.score(x_test, y_test)
# print(f'{score = }')

regr = HistGradientBoostingRegressor()
regr.fit(x_train.reshape(-1, 1), y_train.flatten())

# %%

# pred = rcv.predict(x_test)
pred = regr.predict(x_test)


    

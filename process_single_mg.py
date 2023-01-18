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
# from modules import ajh_utils as util
from modules.ajh_utils import lineplots as lplts
from modules.ajh_utils import handy_dandy as util
import matplotlib.pyplot as plt
import numpy as np

# %%

FILE_DIR = './datasets/MG/'
FILENAME = 'MG0000n005_024.fits'

sigma = 20
nsigma = 10
radius = 1
fwhm = 10
threshold = 10

file_data = fits.getdata(f'{FILE_DIR}{FILENAME}')
cutouts, headers = util.createCutoutsList(file_data)

lplts.plot_gallery(cutouts, 50, 50, 10, 3)


# %%
## move on to masking the sources

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, detect_threshold
from photutils.utils import circular_footprint


# %%
input_data = cutouts[0]
plt.imshow(input_data)

# %%
sigma_clip = SigmaClip(sigma=sigma, maxiters=10)
threshold = detect_threshold(input_data, nsigma=nsigma, sigma_clip=sigma_clip)
segment_img = detect_sources(input_data, threshold, npixels=5)
footprint = circular_footprint(radius=radius)

copy_data = np.copy(input_data)
# copy_data = util.maskBackground(input_data, 50, sigma = sigma)

# %%
plt.imshow(copy_data)

# %%


mask = segment_img.make_source_mask(footprint=footprint)
copy_data[mask] = np.NaN

%matplotlib inline
plt.imshow(copy_data)
# %%

xvalues, row = lplts.GetNthRow(input_data, 25)
xvalues, col = lplts.GetNthColumn(input_data, 25)

%matplotlib widget
plt.plot(xvalues, row)

##  masked data
xvalues, row = lplts.GetNthRow(copy_data, 25)
xvalues, col = lplts.GetNthColumn(copy_data, 25)

plt.plot(xvalues, row)
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


processAndMask(f'{FILE_DIR}/{FILENAME}')

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

data = fits.getdata(f'{FILE_DIR}/{FILENAME}')
mean, med, std  = sigma_clipped_stats(data, sigma=sigma)

dao = DAOStarFinder(fwhm = 5., threshold=5. * std)
sources = dao(file_data - med)

for col in sources.colnames:
    sources[col].info.format = '%.8g'

print(sources)
# %%
from photutils.segmentation import detect_sources, detect_threshold
from photutils.utils import circular_footprint

sclip = SigmaClip(sigma=3, maxiters=10)
threshold = detect_threshold(data, nsigma=nsigma, sigma_clip=sclip)
seg_img = detect_sources(data, threshold, npixels=5)
fprint = circular_footprint(radius=radius)

mask = seg_img.make_source_mask(footprint=fprint)

copy_data = np.copy(data)
copy_data[mask] = np.NaN

%matplotlib inline
plt.imshow(copy_data)

# %%
## go to each source and make a cutout
training_cutouts = []
testing_cutouts = []
for s in sources:
    x = s['xcentroid']
    y = s['ycentroid']

    ## use the masked data
    masked = Cutout2D(copy_data, (x, y), 50).data
    training_cutouts.append(masked)

    testing = Cutout2D(data, (x, y), 50).data
    testing_cutouts.append(testing)

lplts.plot_gallery(training_cutouts, 50, 50, 10, 3)
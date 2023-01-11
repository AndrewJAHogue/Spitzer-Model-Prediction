
# %%
import os
import sys
import joblib

import numpy as np
from astropy.nddata import Cutout2D
# sys.path.append('F:/Python/modules')
from modules.ajh_utils import lineplots, computer_path

# %%

def get_region_cutouts( x, y, h, w,):
    """Loads the SOFIA and Spitzer datasets and creates a cutout centered at (x,y) of size (h, w) 

    Args:
        x (int): x coordinate to center cutout on
        y (int): y coordinate to center cutout on
        h (int): heighth of returned cutouts
        w (int): width of returned cutouts

    Returns:
        Returns two cutouts of equal size, centered on the same (x, y). Returns a cutout from the SOFIA dataset and a cutout from the Spitzer dataset; tuple-like
    """    

    fits_dir = '../Research/fits/Full Maps/Originals/'
    # spits_data = computer_path.Star_Datasets.get_spits_data()
    # sofia_data = computer_path.Star_Datasets.get_sofia_data()
    spits_data = fits.getdata(f'{fits_dir}Spitzer_GCmosaic_24um_onFORCASTheader_JyPix.fits')
    sofia_data = fits.getdata(f'{fits_dir}F0217_FO_IMA_70030015_FORF253_MOS_0001-0348_final_MATT_Corrected.fits')

    spits_cutout = Cutout2D(spits_data, (x, y), (h, w)).data
    sofia_cutout = Cutout2D(sofia_data, (x, y), (h, w)).data

    return sofia_cutout, spits_cutout

    
def rms(y_true, y_pred):
    """Calculate rms error of predictions

    Args:
        y_true (array-like): predictions
        y_pred (array-like): testing data

    Returns:
        ndarray  
    """    

    from numpy import sqrt
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_true, y_pred)
    return sqrt(mse) 


def getSourcesList(input_data, sigma=3.0, fwhm=10., threshold=5.):
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    
    mean, median, std = sigma_clipped_stats(input_data, sigma=sigma, stdfunc=np.nanstd)
    d = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    s = d(input_data - median)
    for col in s.colnames:
        s[col].info.format = "%.8g"

    vals = s['flux']
    mask = vals > std
    s.remove_rows(mask)
    
    print(f'Found {len(s)} stars')

    return s, d


# from photutils.aperture import CircularAperture
# from photutils.detection import DAOStarFinder
# from astropy.visualization.mpl_normalize import ImageNormalize
# from astropy.visualization import SqrtStretch
# from  photutils.aperture import ApertureStats
# positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
# apertures = CircularAperture(positions, r=10.)
# norm = ImageNormalize(stretch=SqrtStretch())

def createCutoutsList(input_data, filename, **keywargs):
    save_fwhm = keywargs.get('save_fwhm', False)
    cutout_size = keywargs.get('cutout_size', (50, 50))
    threshold = keywargs.get('threshold', 5.)
    sigma = keywargs.get('sigma', 3.)
    fwhm = keywargs.get('fwhm', 10.)
    auto_filter = keywargs.get('auto_filter', False)
    
    from astropy.stats import sigma_clipped_stats
    
    sources_list, dao = getSourcesList(input_data, sigma=sigma, fwhm=fwhm, threshold=threshold)
    stats = sigma_clipped_stats(input_data, sigma=sigma, stdfunc=np.nanstd)


    x = sources_list["xcentroid"]
    y = sources_list["ycentroid"]
    points = list(zip(x, y))
    cutouts = []
    cutouts_headers = []

    for p_index, point in enumerate(points):
        
        c = Cutout2D(input_data, point, cutout_size)
        
        ## a filter for any point source that exceeds the std of the mosaic
        mean = np.nanmean(c.data)
        if mean > (stats[0] + stats[2]) and auto_filter:
            continue 


        # save fwhm of original cutout
        if save_fwhm:
            try:
                fwhm = get_fwhm(c.data)
            except ValueError:
                fwhm = "error"
            
            save_fwhm_to_file(fwhm, point, filename)

        if c.shape >= cutout_size:
            cutouts.append(c.data)
            cutouts_headers.append(c)


    

    cutouts = np.array(cutouts)
    cutouts_headers = np.array(cutouts_headers)

    print(f'Created List of Cutouts with size of {len(cutouts)}')
    
    return cutouts, cutouts_headers

def saveCutoutsHeaders(cutouts_headers, filename):
    with open(f'./datasets/cutouts/{filename}_cutouts_headers.jbl', 'wb') as f:
        joblib.dump(cutouts_headers, f)

CUTOUT_SIZE = 50
def maskBackground(input_data, CUTOUT_SIZE=CUTOUT_SIZE, sigma=6.0):
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    sigma_clip = SigmaClip(sigma=sigma)

    bkg_estimator = MedianBackground()

    bkg = Background2D(
        input_data, CUTOUT_SIZE, filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator
    )

    print(f"{bkg.background_median = }\n{bkg.background_rms_median = }")

    input_data_masked = input_data - bkg.background

    # source_mask = mask_sources(input_data, sigma)
    input_data_double_masked = input_data_masked

    return input_data_double_masked


def mask_sources(input_data, sigma, nsigma, radius = 8):
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import detect_sources, detect_threshold
    from photutils.utils import circular_footprint

    sigma_clip = SigmaClip(sigma=sigma, maxiters=10)
    threshold = detect_threshold(input_data, nsigma=nsigma, sigma_clip=sigma_clip)
    segment_img = detect_sources(input_data, threshold, npixels=5)
    footprint = circular_footprint(radius=radius)
    mask = segment_img.make_source_mask(footprint=footprint)
    return mask


def processData(input_data, sigma=3.):
    """
    Description:
        Mask the background noise and impute any nan values 
    Arguments:
        input_data: Must be a 2D array
    Returns:
        Numpy array
    """    


    # masking the background
    masked = maskBackground(input_data, CUTOUT_SIZE, sigma)

    # imputed nan values
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(missing_values=np.NaN, n_neighbors=40)
    imputed_data = imputer.fit_transform(masked)

    return imputed_data



def calc_fwhm(cutout_data):
    import numpy as np
    from scipy.interpolate import CubicSpline, PPoly, UnivariateSpline, splrep
    from specutils.analysis import gaussian_fwhm

    # cutout_data = Cutout2D(data, star_coord, cutout_size).data
    cutout_size = cutout_data.shape[0]
    
    light_profile = np.take(cutout_data, int(cutout_size / 2), axis=0)
    half_max = np.nanmax(cutout_data) / 2
    x = np.linspace(0, len(light_profile), len(light_profile))
    spline = UnivariateSpline(x, light_profile - half_max, s=0)
    # spline = splrep(x, light_profile - half_max, s=0)
    # p = PPoly.from_spline(spline)
    r1, r2 = spline.roots()
    fwhm = r2 - r1
    return fwhm

def save_fwhm_to_file(fwhm, point, filename):
    # import json
    from os.path import isfile

    import joblib

    if '.fits' in filename:
        filename = filename[:len(filename) - 5]

    FILE = f'./datasets/fwhms/{filename}_fwhm.joblib'

    
    # check if file exists
    if isfile(FILE) == False:
        with open(FILE, 'xb') as f:
            pass
        
    fwhm_list = list([ {'fwhm': fwhm, 'coordinate': point } ])

    # print(f'{fwhm_list = }')
    
    with open(FILE, 'wb') as f:
        joblib.dump(fwhm_list, f)    



def twoD_GaussianScaledAmp(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
# Compute FWHM(x,y) using 2D Gaussian fit, min-square optimization
# Optimization fits 2D gaussian: center, sigmas, baseline and amplitude
# works best if there is only one blob and it is close to the image center.
# author: Nikita Vladimirov @nvladimus (2018).
# based on code example: https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m

    import numpy as np
    (x, y) = xy
    xo = float(xo)
    yo = float(yo)    
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns: 
        FWHMs in pixels, along x and y axes.
    """
    import numpy as np
    import scipy.optimize as opt
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    initial_guess = (img.shape[1]/2,img.shape[0]/2,10,10,1,0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, ( x, y ), 
                               img_scaled.ravel(), p0=initial_guess,
                               bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1),
                               (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    return (FWHM_x, FWHM_y)

def get_fwhm(cutout):
    try:
        fwhm_maj, fwhm_min = getFWHM_GaussianFitScaledAmp(cutout)
        fwhm = np.sqrt(fwhm_maj**2 + fwhm_min**2)
        return fwhm
    except RuntimeError:
        if np.any(np.isnan(cutout)):
            print('WARNING: This cutout contains invalid values')

def saveFWHMFile(data, filename):
    import joblib
    with open(f'./datasets/{filename}.fits', 'wb') as f:
        joblib.dump(data, f)

        
def FilterStarsByStd(cutouts, stats):
    filtered_cutouts = []
    for c in cutouts:
        mean = np.nanmean(c)
        if mean < (stats[0] + stats[2]):
            filtered_cutouts.append(c)
        
    # if len(filtered_cutouts) == 1:
    #     filtered_cutouts = filtered_cutouts[0]

    print(f'Filtered down to {len(filtered_cutouts)} stars')
    
    return filtered_cutouts

# %%
def main():
    from astropy.io import fits
    mg610p005 = fits.getdata('./datasets/MG0610p005_024.fits')

    createCutoutsList(mg610p005)
if __name__ == "__main__":
    main()


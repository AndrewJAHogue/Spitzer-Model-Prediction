import numpy as np

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

def Filter(training, testing,  **keywargs):
    """ Filter out cutouts that exceed the mean of the whole training set

    Args:
        input_data (tuple of two ndarrays): (input_training_set, input_testing_set) is a tuple of the training and testing data sets, which are both ndarrays containing (50,50) cutouts

        **keywargs:
            std_coefficient (float, optional): The coefficient to multiply the standard deviation of the whole training set by. Defaults to 1.5.
            sigma (float, optional): The sigma value used for sigma_clipped_stats. Defaults to 3..
            derivfilterfunc (function)

    Returns:
        tuple of two ndarrays: A tuple of (training_set, testing_set) that have been filtered
    """    

    derivfilterfunc = keywargs.get('derivfilterfunc')
    std_coefficient = keywargs.get('std_coefficient', 1.)
    sigma = keywargs.get('sigma', 0.)


    from astropy.stats import sigma_clipped_stats
    import contextlib

    ## optional argument to perform a differential on the input data
    ## otherwise, just keep the data untouched
    if derivfilterfunc is None:
        derivfilterfunc = lambda a : a

    # filter out bad cutouts
    if sigma == 0:
        mean, med, std = sigma_clipped_stats(derivfilterfunc( training ), stdfunc=np.nanstd)
    else:
        mean, med, std = sigma_clipped_stats(derivfilterfunc( training ), sigma=sigma, stdfunc=np.nanstd)

    filtered_training = []
    filtered_testing = []
    for i, c in enumerate(training):
        c_mean = np.nanmean(derivfilterfunc( c ) - med)

        if derivfilterfunc is None and c_mean <= (mean - (std * std_coefficient)) or derivfilterfunc is not None and c_mean >= (mean - (std * std_coefficient)):
            with contextlib.suppress(IndexError):
                filtered_training.append(c)
                filtered_testing.append(testing[i])
    ## turn them from lists to np arrays
    filtered_training = np.array(filtered_training)
    filtered_training = filtered_training.reshape(-1, 2500)

    filtered_testing = np.array(filtered_testing)
    filtered_testing = filtered_testing.reshape(-1, 2500)

    return filtered_training, filtered_testing

def FilterFirstDerivative(training, testing, std_coefficient=1.5, sigma=3.):
    """ Filter out cutouts that exceed the mean of the first derivative whole training set

    Args:
        training (ndarray): (input_training_set, input_testing_set) is a tuple of the training and testing data sets, which are both ndarrays containing (50,50) cutouts
        std_coefficient (float, optional): The coefficient to multiply the standard deviation of the whole training set by. Defaults to 1.5.
        sigma (float, optional): The sigma value used for sigma_clipped_stats. Defaults to 3..

    Returns:
        tuple of two ndarrays: A tuple of (training_set, testing_set) that have been filtered
    """    

    from astropy.stats import sigma_clipped_stats
    import contextlib


    # filter out bad cutouts
    if sigma == 0:
        mean, med, std = sigma_clipped_stats(FirstDerivative(training), stdfunc=np.nanstd)
    else:
        mean, med, std = sigma_clipped_stats(FirstDerivative(training), sigma=sigma, stdfunc=np.nanstd)



    filtered_training = []
    filtered_testing = []
    for i, c in enumerate(training):
        c_diff = (FirstDerivative(c))
        c_mean = np.nanmean(c_diff - med)
        if c_mean >= (mean - ( std * std_coefficient)):
            with contextlib.suppress(IndexError):
                filtered_training.append(c)
                filtered_testing.append(testing[i])

    ## turn them from lists to np arrays
    filtered_training = np.array(filtered_training)
    filtered_training = filtered_training.reshape(-1, 2500)

    filtered_testing = np.array(filtered_testing)
    filtered_testing = filtered_testing.reshape(-1, 2500)
    return filtered_training, filtered_testing




def invertPixelIndex(ypixel, ymax):
    return ymax - ypixel

def getGalDegree(xpixel, ypixel, fits_file):
    from astropy.wcs import WCS

    wcs = WCS(fits_file[0].header)
    deg = wcs.pixel_to_world(xpixel, ypixel).galactic
    # xpixel coord does not matter here, as we only care about the y direction in this case
    return ( deg.l.value, deg.b.value )
    

def getYGalDegree(ypixel, fits_file):
    # xpixel coord does not matter here, as we only care about the y direction in this case
    return getGalDegree(0, ypixel, fits_file)[1]

def isRightSideUp(fits_file):
    """ Check if the 'top' of the image's yaxis (axis=0) is the GC, otherwise return an 
    image ndarray with a flipped yaxis
    

    Args:
        fits_file (HDU object): an HDU object returned from fits.open
    """
    beginning = getYGalDegree(0, fits_file)
    end = getYGalDegree(fits_file[0].data.shape[1], fits_file)

    return beginning**2 > end**2


def isYpixNearGalCenter(ypixel, fits_file):
    b = getYGalDegree(ypixel, fits_file)**2


    ## based on pixel value 2000 for fits_file MG0000n005_025.fits
    return b < .0721377

def isFileWithoutGC(fits_file):
    a = isYpixNearGalCenter(0, fits_file)
    b = isYpixNearGalCenter(fits_file[0].data.shape[1]//2, fits_file)
    c = isYpixNearGalCenter(fits_file[0].data.shape[1], fits_file)

    if a == False and b == False and c == False:
        return True

    
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

    for y in range(yhalf, ymax, 50):
        rightsideuup = isRightSideUp(fits_file)
        if not rightsideuup:
            y = invertPixelIndex(y, ymax)

        if isYpixNearGalCenter(y, fits_file):
            return y

    raise Exception("No Y-Max found for this HDU object")

def sliceImageProperly(fits_file):
    file_data = fits_file[0].data

    if isFileWithoutGC(fits_file):
        return file_data

    if isRightSideUp(fits_file):
        return file_data[:getYMax(fits_file)]
    else:
        return file_data[getYMax(fits_file):]
    



## alternate imputation method
def SimpleImpute(input_data):
    from numpy import isnan

    input_data[isnan(input_data)] = -1
    return input_data
    
def SimpleProcessData(input_data, sigma):
    from modules.ajh_utils import handy_dandy as hd
    SimpleImpute(input_data)
    return hd.maskBackground(input_data, (50,50), sigma)


        

class FWHMinfo:
    def __init__(self, l, b, fwhm):
        self.l = l
        self.b = b
        self.fwhm = fwhm
        
    def getData(self):
        return {
            "l": self.l,
            "b" : self.b,
            "fwhm" : self.fwhm 
                }


def getFWHM(t ):
    from modules.ajh_utils import handy_dandy as hd
    
    try:
        f = hd.get_fwhm(t)
    except ValueError:
        f = np.NaN
        
    return f

def getFWHMobj(t, i, dataset, file):
    headers = dataset.headers
    xpixel,ypixel = headers[i].input_position_original
    l,b = getGalDegree(xpixel, ypixel, file)

    return FWHMinfo(l, b, getFWHM(t))


## return fwhm paired with coordinate for every testing cutout
def getFWHMsForFile(fileset, fits_file):

    # print('getFWHMSForFile')
    testing = fileset.testing_set

    fwhm = lambda t, i : getFWHMobj(t, i, fileset, fits_file)

    return [fwhm(t, i) for i, t in enumerate(testing)] , fileset








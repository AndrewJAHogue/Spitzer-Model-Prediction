
import numpy as np


class FileSet:
    def __init__(self, training_testing_headers_fwhms, source_filename, sigma, nsigma, fwhm, threshold, radius):
        """A class to contain all processed cutouts of a single fits file and what values were used to process them

        Args:
            training_testing_headers_fwhms (tuple of four ndarray): tuple of (training, testing, headers, fwhms), where the training list is a list containing many (50, 50) lists that will be used as the training data
            sigma (float): sigma value used in the DAOStarFinder algorithm to find the stars in the file
            fwhm (float): fwhm value used in the DAOStarFinder algorithm to find the stars in the file
            threshold (float): filter results beyond this limit
        """        
        from modules.ajh_utils import handy_dandy as hd

        self.training_set = training_testing_headers_fwhms[0]
        self.testing_set = training_testing_headers_fwhms[1]
        self.headers = training_testing_headers_fwhms[2]
        self.fwhms = training_testing_headers_fwhms[3]
        self.sigma = sigma
        self.nsigma = nsigma
        self.radius = radius
        self.fwhm = fwhm
        self.threshold = threshold
        self.source_filename = hd.getFileName( source_filename )

        from datetime import datetime
        dt = datetime.now()
        time_str = dt.strftime('%c')

        self.date_created = time_str
        self.date_modified = [ time_str ]

        # self.saved_filename = ''

    def getData(self):
        return np.copy(( self.training_set, self.testing_set ))

    # class training:
    #     def __init__(self):
    #         pass

    # class

    def getFWHMS(self):
        return [f.fwhm for f in self.fwhm]

    def saveFileSet(self):
        import dill
        from modules.ajh_utils import handy_dandy as hd

        self.UpdateFileSetTime()
        
        output_filename = f'{ hd.getFileName(self.source_filename) }_training_testing_headers.dill'
        with open(f'./datasets/cutouts/{output_filename}', 'wb') as f:
            dill.dump(self, f, byref=True)

        # self.saved_filename = output_filename

    def plotAllTesting(self):
        from modules.ajh_utils import lineplots as lplts

        lplts.plot_gallery(self.testing_set, 50, 50, 10, 3)

    def plotAllTraining(self):
        from modules.ajh_utils import lineplots as lplts

        lplts.plot_gallery(self.training_set, 50, 50, 10, 3)
       
        
               
    def plotFWHMHist(self):
        import pandas as pd

        pd.DataFrame(self.getFWHMS()).hist()
       
    def UpdateFileSetTime(self):
        from datetime import datetime
        dt = datetime.now()
        self.date_modified.insert(0, dt.strftime('%c'))

    

        

def CreateFileSet(filepath_or_data,filename, **keywargs):
    """ Automatically create a masked cutout lists from that image, and create a FileSet object from a fits file image. 

    Args:
        filepath (string): string of filepath to fits image

    Returns:
        _type_: _description_
    """    
    sigma = keywargs.get('sigma', 3.)
    nsigma = keywargs.get('nsigma', 10.)
    radius = keywargs.get('radius', 8.)
    fwhm = keywargs.get('fwhm', 10.)
    threshold = keywargs.get('threshold', 5.)
    radius = keywargs.get('radius', 1.)
    peak_percentage = keywargs.get('peak_percentage', 0.7)

    from astropy.io.fits import getdata
    from modules.ajh_utils import handy_dandy as hd
    import numpy as np


    if type(filepath_or_data) == type(str()):
        print('it is a string')
    #     ## only run this block if the argument is a filepath string
        fits_file = fits.open(filepath_or_data)
        file_data = fits_file[0].data


        print(f'{fits_file = }')
        print(f'{file_data.shape = }')

    #     ## run the image data through the function, to slice off the GC portion
    #     # takes the fits file object
        file_data = fp.sliceImageProperly(fits_file)
    else:
        file_data = filepath_or_data

    if filename is None or type(filename) is not str:
        raise ValueError(f"filename {filename} is incorrect or empty")

    training, testing, headers = hd.createMaskedCutoutsList(file_data,
                                                            sigma=sigma,
                                                            nsigma=nsigma,
                                                            fwhm=fwhm,
                                                            threshold=threshold,
                                                            peak_percentage=peak_percentage,
                                                            radius=radius)
    
    ## convert to numpy arrays
    # reshape the training and testing to 2d arrays
    training = np.array(training).reshape(-1, 2500)
    testing = np.array(testing).reshape(-1, 2500)
    headers = np.array(headers)




class MultiSet:
    def __init__(self, filesets, **keywargs):
       self.filename = keywargs.get('filename') 


       from datetime import datetime
       dt = datetime.now()
       time_str = dt.strftime('%c')

       self.date_created = time_str
       self.date_modified = [ time_str ]

       self.source_filesets = filesets

    # TODO fix return methods to numpy.concatenate instead
    def getTrainingData(self):
        return [ f.training_set for f in self.source_filesets]

    def getTestingData(self):
        return [ f.testing_set for f in self.source_filesets]


    def saveMultiSet(self, **keywargs):
        import dill

        filename = keywargs.get('filename')
        if self.filename is None:
            if filename is not None:
                self.filename = filename
            else:
                raise AttributeError("Missing Filename")

        output_filename = f'{self.filename}_multiset.dill'
        with open(f'./datasets/multisets/{output_filename}', 'wb') as f:
            dill.dump(self, f, byref=True)

    
    def UpdateFileSetTime(self):
        from datetime import datetime
        dt = datetime.now()

        self.date_modified.insert(0, dt.strftime('%c'))



    fwhms = []
    # timer.stop()
    print(f'Created FileSet with parameters:\n{sigma = }\n{nsigma = }\n{radius = }\n{fwhm = }\n{threshold = }\nfrom source {filename}')
    return FileSet((training, testing, headers, fwhms), filename, sigma, nsigma, fwhm, threshold, radius)


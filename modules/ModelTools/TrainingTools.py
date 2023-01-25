
import numpy as np


class FileSet:
    def __init__(self, training_testing_headers, sigma, nsigma, fwhm, threshold, radius):
        """A class to contain all processed cutouts of a single fits file and what values were used to process them

        Args:
            training_testing_headers (tuple of three ndarray): tuple of (training, testing, headers), where the training list is a list containing many (50, 50) lists that will be used as the training data
            sigma (float): sigma value used in the DAOStarFinder algorithm to find the stars in the file
            fwhm (float): fwhm value used in the DAOStarFinder algorithm to find the stars in the file
            threshold (float): filter results beyond this limit
        """        
        self.training_set = training_testing_headers[0]
        self.testing_set = training_testing_headers[1]
        self.headers = training_testing_headers[2]
        self.sigma = sigma
        self.nsigma = nsigma
        self.radius = radius
        self.fwhm = fwhm
        self.threshold = threshold

        from datetime import datetime
        dt = datetime.now()
        time_str = dt.strftime(f'%c')
        
        self.date_created = time_str
        self.date_modified = [ time_str ]

        self.saved_filename = ''

    def getData(self):
        return np.copy(( self.training_set, self.testing_set ))

    class training:
        def __init__(self):
            pass

    # class

    def saveFileSet(self, filename):
        import joblib
        from modules.ajh_utils import handy_dandy as hd

        self.UpdateFileSetTime()
        
        output_filename = f'{ hd.getFileName(filename) }_training_testing_headers.jbl'
        with open(f'./datasets/cutouts/{output_filename}', 'wb') as f:
            joblib.dump(self, f)

        self.saved_filename = output_filename

    def plotAllTesting(self):
        from modules.ajh_utils import lineplots as lplts

        lplts.plot_gallery(self.testing_set, 50, 50, 10, 3)

    def plotAllTraining(self):
        from modules.ajh_utils import lineplots as lplts

        lplts.plot_gallery(self.training_set, 50, 50, 10, 3)
       
    def UpdateFileSetTime(self):
        from datetime import datetime
        dt = datetime.now()
        self.date_modified.insert(0, dt.strftime(f'%c'))


class DataSet:
    def __init__(self, FileSetArray):
        self.data = FileSetArray

        from datetime import datetime
        dt = datetime.now()
        self.date_created.append(dt.strftime(f'%c'))
        self.date_modified = [ date_created ]

    def saveDataSet(self, filename):
        import joblib

        # We assume the DataSet is being saved because it was modified
        self.UpdateFileSetTime()
        
        # always use the most recent modified time to save the file
        output_filename = f'{self.date_modified[0]}_training_testing.jbl'
        with open(f'./datasets/training_testing/{output_filename}', 'wb') as f:
            joblib.dump(self, f)

        self.saved_filename = output_filename
        
    def UpdateFileSetTime(self):
        from datetime import datetime
        dt = datetime.now()
        self.date_modified.insert(0, dt.strftime('%c'))

        

def CreateFileSet(filepath, **keywargs):
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

    from astropy.io.fits import getdata
    from modules.ajh_utils import handy_dandy as hd
    import numpy as np


    file_data = getdata(filepath)
    training, testing, headers = hd.createMaskedCutoutsList(file_data, sigma=sigma, nsigma=nsigma, fwhm=fwhm, threshold=threshold, radius=radius)
    
    ## convert to numpy arrays
    # reshape the training and testing to 2d arrays
    training = np.array(training).reshape(-1, 2500)
    testing = np.array(testing).reshape(-1, 2500)
    headers = np.array(headers)

    print(f'Created FileSet with parameters:\n{sigma = }\n{nsigma = }\n{radius = }\n{fwhm = }\n{threshold = }')
    return FileSet((training, testing, headers), sigma, nsigma, fwhm, threshold, radius)


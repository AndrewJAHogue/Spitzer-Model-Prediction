from multiprocessing import pool
from codetiming import Timer
from modules.ajh_utils import handy_dandy as hd
from modules.ProcessingTools import FitsProcessing as fp
import os
import dill

print('Start of file')
# %load_ext autoreload
# %reload_ext autoreload
# %autoreload 2



MIPSGAL_DIR = '../../IRSA_Spitzer_Mipsgal/datasets/'
FILE_DIR = './datasets/MG/'
# FILENAME = 'MG0000n005_024.fits'
# FILENAME = 'MG0000p015_024.fits'


# file = fits.open(f'{FILE_DIR}{FILENAME}')
# dataset = jbl.load(f'./datasets/cutouts/{hd.getFileName(FILENAME)}_training_testing_headers.jbl')
# testing = dataset.testing_set

# print(f'{dataset.source_filename = }')




def getFWHMsForFile(filename):
    from astropy.io import fits
    from modules.ModelTools import TrainingTools as tt

    fits_file =  fits.open(f'{FILE_DIR}{filename}')
    file_data = fits_file[0].data

    sigma = 0.
    fwhm = 10.
    threshold = 10.
    dataset = tt.CreateFileSet(file_data, filename, peak_percentage=0.5, sigma=sigma,fwhm=fwhm, threshold=threshold)
    testing = dataset.testing_set

    fwhm = lambda t, i : fp.getFWHMobj(t, i, dataset, fits_file)

    return [fwhm(t, i) for i, t in enumerate(testing)] , dataset



all_files = os.listdir(FILE_DIR)

if __name__ == '__main__':
    timer = Timer(name="class")
    timer.start()
    
    results = []
    fwhms = []
    with pool.Pool(processes=6) as p:
        print('starting next iteration')
        results = [ p.apply_async(getFWHMsForFile, args=(f,)) for f in all_files ]

        for r in results:
            fwhms, dataset = r.get()
            dataset.fwhm = fwhms
            # dataset.saveFileSet()
            with open(f'./datasets/cutouts/dills/{dataset.source_filename}_training_testing_headers.dill', 'wb') as f:
                dill.dump(dataset, f, byref=True)

            

    timer.stop()

#|%%--%%|


import h5py 
import modules.ModelTools.TrainingTools as tt

set1 = tt.FileSet([[0],[ 1 ],[ 2 ],[ 3 ]], '', 0,0,0,0,0)

def exportFileSetToHDF5(fileset, output_name):
    ## no need to include file extension in output_name
    ## exporting FileSet object to hdf5
    with h5py.File(f'{ output_name }.hdf5', 'w') as f:
        f.create_dataset('training_set', data=fileset.training_set)
        f.create_dataset('testing_set', data=fileset.testing_set)
        # f.create_dataset('headers', data=headers)
        f.create_dataset('fwhms', data=fileset.fwhms)

        f.create_dataset('sigma', data=[fileset.sigma])
        f.create_dataset('nsigma', data=[fileset.nsigma])
        f.create_dataset('radius', data=[fileset.radius])
        f.create_dataset('fwhm', data=[fileset.fwhm])
        f.create_dataset('threshold', data=[fileset.threshold])
        f.create_dataset('source_filename', data=[fileset.source_filename])
        f.create_dataset('date_created', data=[fileset.date_created])
        f.create_dataset('date_modified', data=fileset.date_modified)

def exportCutout2DobjectToHDF5(cutout, output_name):
    with h5py.File(output_name+'.hdf5', 'w') as f:
        c_dict = cutout.__dict__
        for attr in c_dict:
            item = c_dict[attr]
            print(f'{type(item) = }')
            print(f'{item = }')
            ## specifically for 'wcs' attr
            if attr == 'wcs':
                if not item:
                    item = -1
            
            if type(item)  == type(tuple()) and type(item[0]) == slice:
                # slice check
                slices = [sliceAsTuple(sl) for sl in item]
                f.create_dataset(attr, data=[slices])
            else:
                f.create_dataset(attr, data=[item])

        print( f['data'] )

        
        

def sliceAsTuple(slice):
    if not slice.step:
        step = -1
    else:
        step = slice.step
    return slice.start, slice.stop, step

#|%%--%%|
exportFileSetToHDF5(set1, 'filename')
with h5py.File(filename, 'r') as f:
    print(f['training_set'][0])

#|%%--%%|
from astropy.io import fits
from astropy.nddata import Cutout2D
import numpy as np

data = fits.getdata('./datasets/MG/MG0000n005_024 (1).fits')
cutout = Cutout2D(data, (10,10), (50,50))

exportCutout2DobjectToHDF5(cutout, 'cutout')


# with h5py.File('slice.h5', 'w') as f:
#     f.create_dataset('slice', data=np.array(sliceAsTuple( cutout.slices_cutout[0] )))

#|%%--%%|

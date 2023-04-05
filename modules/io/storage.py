#|%%--%%|


import codecs
import h5py 
# import modules.ModelTools.TrainingTools as tt
# import ..ModelTools.TrainingTools as tt
import modules.ModelTools.TrainingTools as tt

dt = h5py.special_dtype(vlen=str)

def exportFileSetToHDF5(fileset, output_name, **keywargs):
    ## no need to include file extension in output_name
    ## exporting FileSet object to hdf5
    # if not folder:
        # folder = h5py.File(f'{ output_name }.hdf5', 'w') 
    # else:
    with h5py.File(f'{ output_name }.hdf5', 'w') as folder:
        if keywargs.get('file'):
            folder = keywargs.get('file')

        folder.create_dataset('training_set', data=fileset.training_set)
        folder.create_dataset('testing_set', data=fileset.testing_set)
        folder.create_dataset('headers', data=cutout_to_bytestring( fileset.headers ), shape=(1,))
        folder.create_dataset('fwhms', data=fileset.fwhms)

        folder.create_dataset('sigma', data=[fileset.sigma])
        folder.create_dataset('nsigma', data=[fileset.nsigma])
        folder.create_dataset('radius', data=[fileset.radius])
        # print(f'{fileset.fwhm = }')
        folder.create_dataset('fwhm', data=cutout_to_bytestring( fileset.fwhm ), shape=(1,))
        folder.create_dataset('threshold', data=[fileset.threshold])
        folder.create_dataset('source_filename', data=[fileset.source_filename])
        folder.create_dataset('date_created', data=[fileset.date_created])
        folder.create_dataset('date_modified', data=fileset.date_modified)

        # folder.close()

def hdf5_to_fileset(folder):
    # with h5py.File(f'{input}', 'r') as folder:
    fileset =  tt.FileSet([folder['training_set'],
                        folder['training_set'],
                        [],
                        folder['fwhms']],
                        folder['source_filename'],
                        folder['sigma'],
                        folder['nsigma'],
                        None,
                        folder['threshold'],
                        folder['radius'],
                        )

    fileset.headers = bytestring_to_cutout(folder['headers'])
    fileset.fwhm = bytestring_to_cutout(folder['fwhm'])
            

def exportMultiSetToHDF5(multiset, output_name):
    with h5py.File(f'{ output_name }.hdf5', 'w') as f:
        f.create_dataset('filename', data=multiset.filename, dtype=dt)
        f.create_dataset('date_created', data=[multiset.date_created])
        f.create_dataset('date_modified', data=multiset.date_modified)

        filesets_grp = f.create_group('source_filesets')
        # [exportFileSetToHDF5(fileset, '', file=fileset_grp) for fileset in multiset.source_filesets]
        for fileset in multiset.source_filesets:
            # print(f'{ fileset.source_filename }')
            grp = filesets_grp.create_group(fileset.source_filename)
            exportFileSetToHDF5(fileset, '', file=grp)

def exportCutout2DobjectToHDF5(cutout, output_name):
    with h5py.File(f'{output_name}.hdf5', 'w') as f:
        c_dict = cutout.__dict__
        for attr in c_dict:
            item = c_dict[attr]
            # print(f'{type(item) = }')
            # print(f'{item = }')
            ## specifically for 'wcs' attr
            if attr == 'wcs' and not item:
                item = -1

            if type(item)  == type(tuple()) and type(item[0]) == slice:
                # slice check
                slices = [sliceAsTuple(sl) for sl in item]
                f.create_dataset(attr, data=[slices])
            else:
                f.create_dataset(attr, data=[item])

        # print( f['data'] )


def import_multiset_from_hdf5(input_hdf5):
    with h5py.File(f'{input_hdf5}.hdf5', 'r') as f:
        sources = []
        for fileset in f['source_filesets']:
            sources.extend(hdf5_to_fileset(fileset))
        
def sliceAsTuple(slice):
    step = slice.step or -1
    return slice.start, slice.stop, step

def cutout_to_bytestring(cutout):
    import dill
    import codecs

    return codecs.encode(dill.dumps(cutout), "base64").decode()

def bytestring_to_cutout(bytestring):
    import dill
    import codecs

    # return dill.loads(codecs.decode(bytestring.encode(), "base64"))
    return dill.loads(codecs.decode(bytestring, "base64"))


#|%%--%%|
if __name__ == '__main__':
    set1 = tt.FileSet([[0],[ 1 ],[ 2 ],[ 3 ]], 'filename1', 0,0,0,0,0)
    set2 = tt.FileSet([[0],[ 1 ],[ 2 ],[ 3 ]], 'filename2', 0,0,0,0,0)
    m1 = tt.MultiSet([set1, set2])

    p = './datasets/hdf5/'
    filename = f'{p}fileset'
    # exportMultiSetToHDF5(m1, './datasets/hdf5/m1')
    
    # with h5py.File(f'{p}m1.hdf5', 'r') as f:
        # for name in f:
        #     print(name)

        # print(f['source_filesets'])
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    import numpy as np

    data = fits.getdata('./datasets/MG/MG0000n005_024 (1).fits')
    cutout = Cutout2D(data, (10,10), (50,50))
    cutout2 = Cutout2D(data, (10,10), (50,50))

    # exportCutout2DobjectToHDF5(cutout, 'cutout')


    with h5py.File('./datasets/hdf5/header.h5', 'w') as f:
        f.create_dataset('header', data=[ cutout_to_bytestring([ cutout , cutout2]) ])

    with h5py.File('./datasets/hdf5/header.h5', 'r') as f:
        cs = bytestring_to_cutout( f['header'][0] )
        print(cs[0].data == cutout.data)

    #|%%--%%|


    import lzma
    import dill
    with lzma.open('./datasets/multisets/multisetone.dill.xz', 'rb') as f:
        multiset = dill.load(f)

    #|%%--%%|

    exportMultiSetToHDF5(multiset, './datasets/hdf5/m1')
    
    #|%%--%%|
    
    with h5py.File(f'./datasets/hdf5/m1.hdf5', 'r+') as f:
        sources = f['source_filesets']
        for i, key in enumerate( sources ):
            source = sources[key]
            del source['headers']
            headers = cutout_to_bytestring( multiset.source_filesets[i].headers )
            # print(f'{len(multiset.source_filesets[i].headers) = }')
            source.create_dataset('headers', data=headers, shape=(1,))
            del headers

            # break
            # source.create_dataset('headers', data=cutout_to_bytestring( multiset.source_filesets[i].headers ))
            # source['headers']  = cutout_to_bytestring( multiset.source_filesets[i].headers )
            # source
            # hdf5_to_fileset(source)
        # hdf5_to_fileset()


    # %%


    with h5py.File(f'./datasets/hdf5/m1.hdf5', 'r') as f:
        sources = f['source_filesets']
        for i, key in enumerate( sources ):
            source = sources[key]
            print(bytestring_to_cutout( source['headers'][0] ))

            break
            # print(( bytestring_to_cutout(  source['headers']  ) ))
            # print(bytestring_to_cutout( source['headers'][0] ))

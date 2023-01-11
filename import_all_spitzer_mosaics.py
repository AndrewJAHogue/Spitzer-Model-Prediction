# %%
import os

from matplotlib.colors import make_norm_from_scale


from tqdm import tqdm


from astropy.stats import sigma_clipped_stats

from modules import handy_dandy as hd

import joblib

import numpy as np

from astropy.io import fits

from multiprocessing import Pool, cpu_count
import pandas as pd

import matplotlib.pyplot as plt

from modules.ajh_utils import lineplots as lplts

from astropy.nddata import Cutout2D


# dir = 'C:/Users/ahogue5/Downloads'
data_dir = './datasets/MG/'
files = os.listdir(data_dir)

    


#%%

def process(file):
    from os.path import isfile

    import joblib

    data = fits.getdata(f'{data_dir}/{file}')

    filename = file[:len(file) - 5]

    # if ( isfile( f'./datasets/fwhms/{filename}_fwhm.joblib' ) == False ) or ( isfile(f'./datasets/cutouts/{filename}_cutouts_headers.jbl') == False ):

    try:

        cutouts, headers = hd.createCutoutsList(data, filename, save_fwhm=False, threshold=5., sigma=10.)

        with open(f'./datasets/cutouts/{filename}_cutouts_headers.jbl', 'wb') as f:

            joblib.dump((cutouts, headers), f)


    except RuntimeError:

        print(f'{filename} failed to generate fwhm list')


# %%


file = files[10]

data = fits.getdata(f'{data_dir}/{file}')

# data = data[:2500, :]


# %%

filename = file[:len(file) - 5]

s = 100. ## the higher the sigma, the more the stars



stats = sigma_clipped_stats(data, sigma=s)

print(f'{stats = }')


t = stats[0] - stats[2] ## the higher the threshold, the fewer the stars

t /= 2

fwhm = 5. ## the higher the fwhm, the more the stars


# %%



cutouts, headers = hd.createCutoutsList(data,
                                        filename,

                                        save_fwhm=True,
                                        threshold=t,

                                        sigma=s,

                                        fwhm=fwhm,

                                        auto_filter=True)


from modules.ajh_utils import lineplots as lplts

lplts.plot_gallery(cutouts, 50, 50, 10, 3)


# hd.saveCutoutsHeaders([cutouts, headers], filename)


base_cutouts = cutouts


# %%

## examining the stats of the cutouts and the mosaic


print(f'Whole mosaic stats = {stats}')

print(f'mean - median = {stats[0] - stats[1]}')



filtered = []

for c in cutouts:
    mean = np.nanmean(c)
    median = np.nanmedian(c)
    std = np.nanstd(c)

    if std > np.round(stats[2], decimals=2):

        print(f'{( mean, std ) = }')
        filtered.append(c)


lplts.plot_gallery(filtered, 50, 50, 10, 3)


# %%

masked_hd = np.copy(headers[0].data)

masked_zoom = Cutout2D(masked_hd, (25, 25), (25,25))

mask = hd.mask_sources(masked_zoom, 10, 20, 1)


# plt.imshow(cutouts[0])

masked_zoom[mask] = np.NaN

plt.imshow(masked_zoom )



# %%

## attempting to auto choose values for sigma and threshold

## don't really like the results so far


# chosen_values = {}


# stats = sigma_clipped_stats(data, sigma=s)

# cutouts_mean_min = np.nanmean(base_cutouts)

# num_samples_max = len(base_cutouts)

# min_std = stats[2]

# for sigma in np.linspace(5, 20, 5):

#     for threshold in np.linspace(5, 20, 5):

#         cutouts, headers = hd.createCutoutsList(data,
#                                                 filename,

#                                                 save_fwhm=False,
#                                                 threshold=threshold,

#                                                 sigma=sigma,

#                                                 fwhm=fwhm,

#                                                 auto_filter=True)

#         cutouts_mean = np.nanmean(cutouts)
#         num_samples = len(cutouts)
        

#         if cutouts_mean_min > cutouts_mean and num_samples_max < num_samples:

#             # if np.nanstd(cutouts) < np.round(std, decimals=2):

#             chosen_values['sigma'] = sigma

#             chosen_values['threshold'] = threshold 

#             print(f'Found new values = {chosen_values}')

            

# print(chosen_values) 



# %%

# import dill

# import tqdm

# with Pool() as p:

#     with Pool(processes=cpu_count()-2) as p:

#         a = list(tqdm.tqdm(  p.imap_unordered(process, files), total=1 ))

#         # a =  p.imap_unordered(process, 'MG0010p005_024.fits')


#         for i in a:

#             if i:
#                 print(i)

        # for index, i in enumerate(a):

        #     if i:

        #         file_name = i[0][:14]

        #         with open(f'./datasets/mass processing/{file_name}_processed.jbl', 'wb') as f:

        #             dill.dump(i[1], f)

        #         with open(f'./datasets/mass processing/{file_name}_masked.jbl', 'wb') as f:

        #             joblib.dump(i[2], f)


# %%

# n005 = joblib.load('./fwhms/MG0000n005_024 (1).fits_fwhm.joblib')


# %%

def processFWHMList(jbl):

    fwhms = []

    coords = []

    for i, item in enumerate( jbl ):

        item_dict = item[0]

        try:

            fwhms.append(float( item_dict['fwhm'] ))

        except ValueError:

            fwhms.append(item_dict['fwhm'])

        coords.append(item_dict['coordinate'])

    import pandas as pd

    import numpy as np

    return pd.DataFrame(np.array( [fwhms, coords] ).T, columns=['fwhm', 'coord'])


# %%

# process all generated jbls

# dir = './fwhms'
# files = os.listdir(dir)

# for jbl in files:

#     name = jbl[:len(jbl) - 7]

#     jbl_data = joblib.load(f'{dir}/{jbl}') 

#     with open(f'./datasets/dataframes/{name}_dataframe.jbl', 'wb') as f:

#         df = processFWHMList(jbl_data)

#         joblib.dump(df, f)


jbl = joblib.load(f'./datasets/cutouts/{filename}_cutouts_headers.jbl')



# %%


# n005_df = processFWHMList(n005)


def removeErrors(df):

    fwhms = df.fwhm

    return fwhms[fwhms != 'error']


def histogram(df):

    fwhms = removeErrors(df)

    fwhms.plot.hist()



# # %%

# # combine all dfs

# dir = './datasets/dataframes/'
# files = os.listdir(dir)


# df1 = joblib.load(dir + files[0])

# for i, file in enumerate( files ):

#     if (i + 1) < len(files):

#         df2 = joblib.load(dir + files[i + 1])

#         df1.merge(df2, how='right')


# %%

# all_fwhms = removeErrors(df1)

# all_fwhms.plot.hist()# 
"""
    A worked example of the segmentation section on PhotUtils docs
"""
# %%
from photutils.background import Background2D, MeanBackground
from photutils.segmentation import make_2dgaussian_kernel, detect_sources
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# %%
FILE_DIR = './datasets/MG/'
FILENAME = 'MG0000n005_024.fits'

file_data = fits.getdata(f'{FILE_DIR}{FILENAME}')

# %%

bkg_estimator = MeanBackground()
bkg = Background2D(file_data, (50, 50), filter_size=(3,3), bkg_estimator=bkg_estimator)
file_data -= bkg.background

threshold = 1.5 * bkg.background_rms

# %%

seg_map = detect_sources(file_data, threshold, npixels=10, connectivity=8)
plt.imshow(seg_map)

# %%import numpy as np

import matplotlib.pyplot as plt

from astropy.visualization import SqrtStretch

from astropy.visualization.mpl_normalize import ImageNormalize

norm = ImageNormalize(stretch=SqrtStretch())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))

ax1.imshow(file_data, origin='lower', cmap='Greys_r', norm=norm)

ax1.set_title('Background-subtracted Data')

ax2.imshow(seg_map, origin='lower', cmap=seg_map.cmap,

           interpolation='nearest')

ax2.set_title('Segmentation Image')


# %%

## so, this produces a very garbled mess

from photutils.segmentation import SourceFinder
finder = SourceFinder(npixels=6, progress_bar=False)
seg_map = finder(file_data, threshold)
plt.imshow(seg_map)
# %%

# %%

## lets try DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

mean, med, std = sigma_clipped_stats(file_data, sigma=3.)

dao = DAOStarFinder(fwhm=3., threshold=5.*std)
sources = dao(file_data - med)

for col in sources.colnames:
    sources[col].info.format = '%8g'

print(sources)

# %%
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture

positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(file_data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
apertures.plot(color='blue', lw=1.5, alpha=0.5)

# %%
## masking near the GC
mask = np.zeros(file_data.shape, dtype=bool)
mask[2500:, :] = True

sources = dao(file_data - med, mask=mask)
for col in sources.colnames:
    sources[col].info.format = '%.8g'


positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(file_data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
apertures.plot(color='blue', lw=1.5, alpha=0.5)


# %%

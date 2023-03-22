# %%

# %load_ext autoreload
# # %reload_ext autoreload
# %autoreload 2
import importlib
import os

import dill
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from modules.ajh_utils import handy_dandy as hd
from modules.ModelTools import TrainingTools as tt

importlib.reload(tt)
from modules.ProcessingTools import FitsProcessing as fp

# %%


with open("./datasets/models/knnOne_training_testing.dill", "rb") as f:
    knn, filtered_train, filtered_test = dill.load(f)


# %%


if len(filtered_train):

    x_train, x_test, y_train, y_test = train_test_split(filtered_train, filtered_test, test_size=0.2)

    knn = KNeighborsRegressor()
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print(f"{score = }")

    pred = knn.predict(x_test)
    # lplts.plot_gallery(pred, 50, 50, 30, 4, stats=True)

    # %%

    tt.comparePredictions(x_test, y_test, pred)

    # %%
    ## this is where we import our main spitzer data, take some cutouts, and see if our model can predict them well enough
    
    ## import spits_iso1 
    from astropy.io import fits
    from modules.ajh_utils import computer_path as cp
    from astropy.nddata import Cutout2D
    import matplotlib.pyplot as plt

    spits_iso1 = fits.getdata(cp.spits_iso1())
    sofia_iso1 = fits.getdata(cp.sofia_iso1())
    coords = (276, 247)

    spits1 = Cutout2D(spits_iso1, coords, (50, 50) ).data
    sofia1 = Cutout2D(sofia_iso1, coords, (50, 50) ).data

    training = fp.SimpleProcessData(spits1, 0)
    training = [ training.flatten() ]

    testing = sofia1

    pred_spits = knn.predict(training)

    
    plt.subplot(131)
    plt.imshow(training[0].reshape(50,50))
    plt.subplot(132)
    plt.imshow(testing)
    plt.subplot(133)
    plt.imshow(pred_spits[0].reshape(50,50))



## our model doesn't really handle the exotic shapes that well, even if they are pretty Gaussian

# %%

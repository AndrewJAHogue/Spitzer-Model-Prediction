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


with open('./datasets/models/knnOne_training_testing.dill', 'rb') as f:
    knn, filtered_train, filtered_test = dill.load(f)


# %%


if len(filtered_train):

    # filtered_train = np.array(filtered_train)
    # filtered_test = np.array(filtered_test)
    
    x_train, x_test, y_train, y_test = train_test_split(
        filtered_train, filtered_test, test_size=0.2
    )


    knn = KNeighborsRegressor()
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print(f'{score = }')






    pred = knn.predict(x_test)
    # lplts.plot_gallery(pred, 50, 50, 30, 4, stats=True)
    
    # %%
    
    tt.comparePredictions(x_test, y_test, pred)
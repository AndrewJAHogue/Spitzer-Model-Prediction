# %%
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
# from ..modules.ajh_utils.lineplots import compare_results, plot_gallery
from modules.ajh_utils import lineplots
import modules.handy_dandy as hd

dir = './datasets/testing data'
# training_file = './datasets/training data/day30-month09_time-13-03_training.jbl'
# testing_file = './datasets/testing data/day30-month09_time-14-00_testing.jbl'
training_file = './knn_training_data_masked.joblib'
testing_file = './knn_training_data.joblib'

training_data = joblib.load(training_file)
testing_data = joblib.load(testing_file)

# %%
lineplots.plot_gallery(training_data, 50, 50, 10, 3)

# %%
# split the data
input_train, input_test, output_train, output_test = train_test_split(
    training_data.reshape(-1, 2500), testing_data, test_size=0.2
    )

rcv = RidgeCV()
rcv.fit(input_train, output_train)
# %%

prediction = rcv.predict(input_test)
lineplots.plot_gallery(prediction, 50, 50, 10, 3)







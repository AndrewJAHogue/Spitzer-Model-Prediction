# %%
# imports
import time
import joblib
import numpy as np
from matplotlib import pyplot as plt
from pexpect import EOF
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from modules import handy_dandy as hd
from astropy.io import fits
import os
from modules.ajh_utils import lineplots



def reshape_to_2dList(cutouts):
    return cutouts.reshape(-1, 2500)

dir = './datasets/cutouts'
files = os.listdir(dir)

# %%


training_data = []
testing_data = []
# for file in ( files ):

def do_all(file):
    if 'headers' in file:
        time.sleep(1)
        try:
            print(f'{dir}/{file} loaded')
            data, headers = joblib.load(f'{dir}/{file}')
            processed_data = []
            masked_data = []
            processed_data = hd.processData(data.reshape(-1,1) )
            processed_data = processed_data.reshape(-1, 50, 50)

            for c in processed_data:
                try:
                    masked_data.append(hd.mask_sources(c, 10, 6))
                except AttributeError:
                    masked_data.append(hd.mask_sources(c, 1, 6))

            masked_data = np.array(masked_data)


            return (file, processed_data , masked_data)
        except:
            print(f'{dir}/{file} failed')
            
        
            

# %%
def dumpAll():
    from multiprocessing import Pool, cpu_count
    import dill
    import tqdm
    with Pool(processes=cpu_count()-4) as p:
        a = list(tqdm.tqdm(  p.imap_unordered(do_all, files), total=len(files)  ))

        for index, i in enumerate(a):
            if i:
                file_name = i[0][:14]
                with open(f'./datasets/mass processing/{file_name}_processed.jbl', 'wb') as f:
                    dill.dump(i[1], f)
                with open(f'./datasets/mass processing/{file_name}_masked.jbl', 'wb') as f:
                    joblib.dump(i[2], f)


# %%
dir = './datasets/MG mass processing/'
files = os.listdir(dir)
file_testing = files[0]
file_training = files[1]
# training_data = np.array([])
# testing_data = np.array([])
# for file in files:
#     try:
#         data = joblib.load(dir + file) 

#         if 'processed' in file:
#             testing_data = np.append(testing_data, data)
#         elif 'masked' in file:
#             training_data = np.append(training_data, data)

#     except PermissionError:
#         pass

training_data = joblib.load(dir + file_training)
testing_data = joblib.load(dir + file_testing)


testing_data = testing_data.reshape(-1, 2500)
training_data = training_data.reshape(-1, 2500)
    
lineplots.plot_gallery(training_data, 50, 50, 10, 3)
lineplots.plot_gallery(testing_data, 50, 50, 10, 3)

print(f'{file_testing, file_training}')

# %%


input_train, input_test, output_train, output_test = train_test_split(
    training_data, testing_data, test_size=0.2, shuffle=False
)

# %%

# setup model
rcv = RidgeCV()
rcv.fit(training_data.reshape(-1, 2500), testing_data.reshape(-1, 2500))

rcv.score(input_test.reshape(-1, 2500), output_test.reshape(-1, 2500))
# predictions
pred = rcv.predict(input_test.reshape(-1, 2500))
pred = pred.reshape(-1, 50, 50)

# %%
# plot the predictions
lineplots.plot_gallery(pred, 50, 50, 10, 3)
lineplots.plot_gallery(training_data, 50, 50, 10, 3)

# rms
print( hd.rms(output_test.reshape(-1, 1), pred.reshape(-1, 1)) )

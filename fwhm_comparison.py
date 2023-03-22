
# %%

import dill
import matplotlib.pyplot as plt

from modules.ajh_utils import handy_dandy as hd
import numpy as np
from modules.ModelTools import TrainingTools as tt
from modules.ProcessingTools import FitsProcessing as fp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsRegressor

# %%

with open('./datasets/multisets/MultiSet_One.dill', 'rb') as f:
    multiset = dill.load(f)

with open("./datasets/models/knnOne_training_testing.dill", "rb") as f:
    knn, filtered_train, filtered_test = dill.load(f)

# %%

# filtered_train = np.array(filtered_train)
# filtered_test = np.array(filtered_test)

x_train, x_test, y_train, y_test = train_test_split(
    filtered_train, filtered_test, test_size=0.2
)


knn = KNeighborsRegressor()
knn.fit(x_train, y_train)
score = knn.score(x_test, y_test)
print(f'{score = }')




# %%

pred = knn.predict(filtered_train)
# lplts.plot_gallery(pred, 50, 50, 30, 4, stats=True)
# plt.show()


# %%


##------------- FWHM comparison------------------------------------------------------------------------------------------------------------------------------------------------

# source_fwhms = [fp.getFWHM(c) for c in y_test]
# # %%

# source_fwhms = np.array(source_fwhms, dtype='float64')
# source_fwhms = source_fwhms[~np.isnan(source_fwhms)]

## attempting getting fwhms with multiprocessing

from multiprocessing import Lock, Manager
from tqdm import tqdm

def mp(cutout, other):
    from modules.ProcessingTools import FitsProcessing as fp
    return fp.getFWHM(cutout), fp.getFWHM(other)

source_fwhms = []
pred_fwhms = []

if __name__ == '__main__':
    import dill
    import os
    import matplotlib.pyplot as plt

    from codetiming import Timer
    from modules.ajh_utils import handy_dandy as hd
    import numpy as np
    from modules.ModelTools import TrainingTools as tt
    from modules.ProcessingTools import FitsProcessing as fp
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import  KNeighborsRegressor

    from pathos.multiprocessing import ProcessingPool as Pool
    from multiprocessing import Manager


    timer = Timer(name='class')
    timer.start()

    with Pool() as p:
        # results = [ p.apipe(mp, c) for c in filtered_test]
        results = [p.apipe(mp, c, pred[i]) for i, c in enumerate( filtered_test )]            

        for r in results:
            s, p = r.get()
            source_fwhms.append(s)
            pred_fwhms.append(p)

    timer.stop()

# %%

new_source_fwhms = []
new_pred_fwhms = []
for i, f in enumerate(source_fwhms):
    if f and pred_fwhms[i]:
        new_source_fwhms.append(f)
        new_pred_fwhms.append(pred_fwhms[i])




# %%

new_source_fwhms = np.array(new_source_fwhms)
new_pred_fwhms = np.array(new_pred_fwhms)



n, bins, patches = plt.hist(x=new_source_fwhms - new_pred_fwhms, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.xlim(3, 8)
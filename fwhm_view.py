#|%%--%%| <sly7MtzfzY|CFI9cZifOU>
import os
import dill

datasets_dir = './datasets/cutouts/dills/'

files = os.listdir(datasets_dir)

file = files[200]
print(f'{file = }')

with open(datasets_dir + file, 'rb') as f:
    dataset = dill.load(f)


dataset.plotFWHMHist()

#|%%--%%| <CFI9cZifOU|DobUSINwQV>


def Train(fits_file):
    from sklearn.model_selection import train_test_split
    from modules.ProcessingTools import FitsProcessing as fp
    from sklearn.neighbors import  KNeighborsRegressor
    from modules.ajh_utils import lineplots as lplts
    import matplotlib.pyplot as plt

    file_data = fits_file[0].data
    # slice it up
    file_data = fp.sliceImageProperly(fits_file)


    sigma = 0.
    # nsigma = 1000.
    fwhm = 10.
    threshold = 10.
    radius = 1
    set1 = tt.CreateFileSet(file_data, peak_percentage=0.5, sigma=sigma,fwhm=fwhm, threshold=threshold)
    training, testing = set1.getData()
    print(f'{training.shape = }')

    filtered_training, filtered_testing = fp.Filter(training, testing, std_coefficient=1, sigma=0)
    print(f'{filtered_training.shape = }')
    filtered_training = fp.SimpleProcessData(filtered_training, sigma)
    filtered_testing = fp.SimpleImpute(filtered_testing)

    x_train, x_test, y_train, y_test = train_test_split(
        filtered_training, filtered_testing, test_size=0.2
    )



    knn = KNeighborsRegressor()
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print(f'{score = }')



    pred = knn.predict(x_test)
    lplts.plot_gallery(pred, 50, 50, 50, 4, stats=True)
    plt.show()

def TrainFileSet(FileSet):
    from sklearn.model_selection import train_test_split
    from modules.ProcessingTools import FitsProcessing as fp
    from sklearn.neighbors import  KNeighborsRegressor
    from modules.ajh_utils import lineplots as lplts
    import matplotlib.pyplot as plt

    # slice it up
    file_data = fp.sliceImageProperly(fits_file)


    sigma = 0.
    # nsigma = 1000.
    fwhm = 10.
    threshold = 10.
    radius = 1
    set1 = tt.CreateFileSet(file_data, peak_percentage=0.5, sigma=sigma,fwhm=fwhm, threshold=threshold)
    training, testing = set1.getData()
    print(f'{training.shape = }')

    filtered_training, filtered_testing = fp.Filter(training, testing, std_coefficient=1, sigma=0)
    print(f'{filtered_training.shape = }')
    filtered_training = fp.SimpleProcessData(filtered_training, sigma)
    filtered_testing = fp.SimpleImpute(filtered_testing)

    x_train, x_test, y_train, y_test = train_test_split(
        filtered_training, filtered_testing, test_size=0.2
    )



    knn = KNeighborsRegressor()
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print(f'{score = }')



    pred = knn.predict(x_test)
    lplts.plot_gallery(pred, 50, 50, 50, 4, stats=True)
    plt.show()


#|%%--%%| <DobUSINwQV|FPnJHWfHkW>


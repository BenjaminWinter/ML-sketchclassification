from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from matplotlib.colors import Normalize
import config
import matplotlib.pyplot as plt
import numpy as np



class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
        
def runAll():

    print("Loading Data...")
    x = np.load(config.FILE_DATA)
    y = np.load(config.FILE_TARGET)
    
    print("Testing...")
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-9, 3, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(decision_function_shape="ovr", kernel="linear"), param_grid=param_grid, cv=2, n_jobs=8, verbose=100)
    grid.fit(x, y)

    print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
        
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
            norm=MidpointNormalize(vmin=0.7, midpoint=0.9, vmax=1.0))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
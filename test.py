import config
import multiprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

def runAll():
    print "Loading Data..."
    x = np.load(config.FILE_DATA)
    y = np.load(config.FILE_TARGET)
    print "Testing..."
    clf = svm.SVC(C=1,decision_function_shape="ovr", kernel="linear", gamma="auto")

    f1 = cross_val_score(clf,x,y,scoring="f1_weighted",cv=10,n_jobs=multiprocessing.cpu_count(), verbose=100)
    print "F1: ",f1
    print "Mean F1: ",f1.mean()
    print "Testing finished OK."
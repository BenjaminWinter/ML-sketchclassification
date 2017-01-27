import config
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

def runAll():
    print "Loading Data..."
    x = np.load(config.FILE_DATA)
    y = np.load(config.FILE_TARGET)
    print "Testing..."
    clf = svm.SVC(C=1,decision_function_shape="ovr", kernel="linear", gamma="auto")

    # precision = cross_val_score(clf,x,y,scoring="precision_weighted",cv=10,n_jobs=4, verbose=1)
    # recall = cross_val_score(clf,x,y,scoring="recall_weighted",cv=10,n_jobs=4, verbose=1)
    f1 = cross_val_score(clf,x,y,scoring="f1_weighted",cv=10,n_jobs=4, verbose=1)
    # print "Mean precision: ", precision.mean()
    # print "Mean recall: ", recall.mean()
    print "F1: ",f1
    print "Mean F1: ",f1.mean()
    print "Testing finished OK."
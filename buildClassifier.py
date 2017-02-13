import config
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

def learn():
    
    print "Loading Data..."
    x = np.load(config.FILE_DATA)
    y = np.load(config.FILE_TARGET)
    
    print "Building Classifier..."
    clf = svm.SVC(decision_function_shape="ovr", kernel="linear", gamma="auto", n_jobs=8, C=0.01)
    clf.fit(x,y)
    joblib.dump(clf,config.CLASSIFIER)
    print "Building Classifier finished OK."


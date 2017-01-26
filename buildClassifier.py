import config
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

def learn():
    print "Building Classifier..."
    x,y = load_svmlight_file(config.DATAFILE, zero_based=True)
    clf = svm.SVC(decision_function_shape="ovo", kernel="rbf", degree="3", gamma="auto")
    clf.fit(x,y)
    joblib.dump(clf,config.CLASSIFIER)
    print "Building Classifier finished OK."


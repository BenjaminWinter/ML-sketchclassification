import config
from sklearn.externals import joblib
import DataPreparation

def getPredictions(imgs):
    clf = joblib.load(config.CLASSIFIER)
    extracted = DataPreparation.multiExtract(imgs)
    predictions = clf.predict(extracted)
    readable = [config.TARGET_MAP_INVERSE[target] for target in predictions]
    print "Predictions: ", readable
    pass
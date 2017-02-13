import config
from sklearn.externals import joblib
import DataPreparation

def getPredictions():
    clf = joblib.load(config.CLASSIFIER)

    while True:
        img = input("Path to Image: ")
        if img == "exit":
            print "Bye"
            exit()

        extracted = DataPreparation.extract(img)
        prediction = clf.predict_proba(extracted)
        readable = config.TARGET_MAP_INVERSE[prediction]
        print "Predictions: ", readable


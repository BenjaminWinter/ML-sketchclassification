import config
from sklearn.externals import joblib
import DataPreparation, numpy as np

def getPredictions():
    print "Load Classifier..."
    clf = joblib.load(config.CLASSIFIER)

    while True:
        img = raw_input("Path to Image: ")
        if img == "exit":
            print "Bye"
            exit()

        print "Extracting Features..."
        extracted = np.array(DataPreparation.extract(img)).reshape((1, -1))
        print "Predicting..."
        predictions = clf.predict_proba(extracted)
        print "---------------"
        print predictions[0]
        print "---------------"
        print sorted(zip(predictions[0], config.TARGET_MAP.keys()))
        exit()
        readable = config.TARGET_MAP_INVERSE[predictions]
        print "Predictions: ", readable

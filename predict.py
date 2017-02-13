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
        tuples = sorted(zip(predictions[0], sorted(config.TARGET_MAP.keys())),reverse=True);
        best = tuples[:5]
        print ""
        print "Top 5 Predictions:"
        print ""
        for i, result in enumerate(best):
            print "%d. %s | Probability: %.1f " % (i+1,result[1],result[0]*100)
        print ""

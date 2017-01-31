import argparse, buildClassifier,predict,test
import DataPreparation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifying sketches')
    parser.add_argument("-l", "--learn", help="train the svm", action="store_true")
    parser.add_argument("-p", "--predict", help="predict an image", action="store", nargs="+")
    parser.add_argument("-d", "--prepare", help="extract features from training set and save", action="store_true")
    parser.add_argument("-t", "--test", help="test the svm", action="store_true")
    parser.add_argument("-r", "--run", help="run through entire workflow", action="store", nargs="+")
    args = parser.parse_args()

    if args.learn:
        buildClassifier.learn()
    elif args.predict:
        predict.getPredictions(args.predict)
    elif args.prepare:
        DataPreparation.ExtractAndSave()
    elif args.test:
        test.runAll()
    elif args.run:
        buildClassifier.learn()
        test.runAll()
        predict.getPredictions(args.predict)
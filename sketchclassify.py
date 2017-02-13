import argparse, buildClassifier,predict,test, grid_search
import DataPreparation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifying sketches')
    parser.add_argument("-l", "--learn", help="train the svm", action="store_true")
    parser.add_argument("-p", "--predict", help="predict an image", action="store_true")
    parser.add_argument("-d", "--prepare", help="extract features from training set and save", action="store_true")
    parser.add_argument("-t", "--test", help="Do Crossvalidation", action="store_true")
    parser.add_argument("-g", "--gridsearch", help="do a gridsearch for parameters", action="store_true")
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
    elif args.gridsearch:
        grid_search.runAll()
    elif args.run:
        buildClassifier.learn()
        test.runAll()
        predict.getPredictions(args.predict)
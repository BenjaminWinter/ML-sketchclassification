import os

FILE_DATA = "./persistence/data.npy"
FILE_TARGET = "./persistence/targets.npy"
IMGDB = "./db/**/*"
CLASSIFIER = "./persistence/classifier.pkl"

dirs = [os.path.basename(x[0]) for x in os.walk("./db")]
dirs.remove('db')
dirs = sorted(dirs)
TARGET_MAP = dict(zip(dirs, xrange(len(dirs))))
TARGET_MAP_INVERSE = {v: k for k, v in TARGET_MAP.items()}

FILE_TARGET = "./persistence/data.npy"
FILE_DATA = "./persistence/targets.npy"
IMGDB = "./db/**/*"
CLASSIFIER = "./persistence/classifier.pkl"
TARGET_MAP = {"apple":0,"bus":1,"calculator":2,
              "envelope":3,"eyeglasses":4,"face":5,
              "head-phones":6,"sailboat":7,"snowman":8,"t-shirt":9}
TARGET_MAP_INVERSE = {v: k for k, v in TARGET_MAP.items()}
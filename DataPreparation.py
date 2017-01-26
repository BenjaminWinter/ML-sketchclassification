import config, cv2, glob, sys, numpy as np
from multiprocessing import Process, Queue
from tqdm import tqdm

def ExtractAndSave():

    save(extractFromDb())
    print "Extract and Save finished OK."

def extractFromDb():
    print "Extracting Features from Database..."
    features= []
    dbimgs = glob.glob(config.IMGDB)
    todo = np.array_split(dbimgs,4)
    resultQ = Queue()
    sift = cv2.xfeatures2d.SIFT_create()
    jobs=[]
    #kps = create_keypoints(1111,1111,30,10)
    kps_low = create_keypoints(1111,1111,300,5)
    kps_mid = create_keypoints(1111,1111,120,10)
    kps_high = create_keypoints(1111,1111,70,20)
    # kps_low = create_keypoints(1111,1111,150,10)
    # kps_mid = create_keypoints(1111,1111,50,20)
    # kps_high = create_keypoints(1111,1111,20,30)
    kps_final = kps_low + kps_mid + kps_high
    for w in xrange(4):
        p = Process(target=extractWorker, args=(todo[w],resultQ,sift, w, kps_final))
        p.start()
        jobs.append(p)

    while True:
        running = any(j.is_alive() for j in jobs)
        while not resultQ.empty():
            features.append(resultQ.get())
        if not running:
            break

    print "\n"*5
    print "Extraction Finished"
    return features

def multiExtract(imgs):
    sift = cv2.xfeatures2d.SIFT_create()
    temp_features = []
    for idx, img in enumerate(imgs):
        temp_features.append(extract(img, sift))
    return temp_features

def extractWorker(todo,done,sift, pnumber, kps):
    for idx, img in enumerate(tqdm(todo,position=pnumber)):
        target = getTarget(img)
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp,des = sift.compute(gray,kps)
        features = np.ravel(des)
        res = np.insert(features,0,target)
        done.put(res)



def extract(img, sift):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)
    return np.ravel(des)


def save(features):

    print "Saving Data..."
    min_length= 999999
    max_length = 0
    with open(config.DATAFILE, 'w') as f:
        for idx, row in enumerate(tqdm(features)):
            line = ""
            if len(row)> max_length:
                max_length = len(row)
            if len(row)< min_length:
                min_length = len(row)
            for i, feature in enumerate(row):
                if i == 0:
                    line += str(int(feature))
                else:
                    line += " " + str(i) + ":" + str(int(feature))
            line += "\n"
            f.write(line)


def getTarget(path):
    parts = path.split("/")
    return config.TARGET_MAP[parts[len(parts)-2]]

def create_keypoints(w, h, size, density):
    keypoints = []
    xoffset =w/density/2
    yoffset = h/density/2
    for x in range(density):
        for y in range(density):
            xpos = x*w/density+xoffset
            ypos = y*w/density+yoffset
            keypoints.append(cv2.KeyPoint(xpos,ypos,size))

    return keypoints
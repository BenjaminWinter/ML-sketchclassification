import config, cv2, glob, sys, os,time,  numpy as np
from multiprocessing import Process, Queue
import multiprocessing
from tqdm import tqdm

def ExtractAndSave():

    f,t = extractFromDb()
    save(f,t)
    print "Extract and Save finished OK."

def extractFromDb():
    print "Extracting Features from Database..."
    features= []
    targets = []
    dbimgs = glob.glob(config.IMGDB)
    todo = np.array_split(dbimgs,multiprocessing.cpu_count())
    resultQ = Queue()
    targetQ = Queue()
    sift = cv2.xfeatures2d.SIFT_create()
    jobs=[]
    #kps_low = create_keypoints(1111,1111,30,5)
    kps_low = create_keypoints(1111,1111,300,5)
    kps_mid = create_keypoints(1111,1111,120,10)
    kps_high = create_keypoints(1111,1111,70,20)
    kps_final = kps_low + kps_mid + kps_high
    #kps_final = kps_low
    for w in xrange(multiprocessing.cpu_count()):
        p = Process(target=extractWorker, args=(todo[w],resultQ,targetQ,w))
        p.start()
        jobs.append(p)

    while True:
        running = any(j.is_alive() for j in jobs)
        while not resultQ.empty():
            features.append(resultQ.get())
        while not targetQ.empty():
            targets.append(targetQ.get())
        if not running:
            break
        time.sleep(0.02)

    print "\n"*5
    print "Extraction Finished"
    return features, targets

def multiExtract(imgs):
    sift = cv2.xfeatures2d.SIFT_create()
    temp_features = []
    for idx, img in enumerate(imgs):
        temp_features.append(extract(img, sift))
    return temp_features

def extractWorker(todo,done,targetQ, pnumber):
    sift = cv2.xfeatures2d.SIFT_create()
    kps_low = create_keypoints(1111,1111,300,5)
    kps_mid = create_keypoints(1111,1111,120,10)
    kps_high = create_keypoints(1111,1111,70,20)
    kps = kps_low + kps_mid + kps_high
    for idx, img in enumerate(tqdm(todo,position=pnumber)):
        target = getTarget(img)
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp,des = sift.compute(gray,kps)
        features = np.ravel(des)
        #res = np.insert(features,0,target)
        done.put(features)
        targetQ.put(target);



def extract(img, sift):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)
    return np.ravel(des)


def save(features, targets):

    print "Saving Data..."
    np.save(config.FILE_DATA, features)
    print "Saving Targets..."
    np.save(config.FILE_TARGET, targets)
    # with open(config.DATAFILE, 'w') as f:
    #     for idx, row in enumerate(tqdm(features)):
    #         line = ""
    #         for i, feature in enumerate(row):
    #             if i == 0:
    #                 line += str(int(feature))
    #             else:
    #                 line += " " + str(i) + ":" + str(int(feature))
    #         line += "\n"
    #         f.write(line)


def getTarget(path):
    target = os.path.split(os.path.split(path)[0])[1]
    return config.TARGET_MAP[target]

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
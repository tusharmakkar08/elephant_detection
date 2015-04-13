import cv2, numpy
from sklearn import svm

def train_positive():
    '''
    Training positive elephant sample
    '''
    hog = cv2.HOGDescriptor()
    inp_pos_train = []
    out_pos_train = []
    for i in xrange(1,3):
        if i<10:
            image_no = "/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalPositive/image_000"+str(i)+".jpg"
        else:
            image_no = "/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalPositive/image_00"+str(i)+".jpg"
        im = cv2.imread(image_no)
        h = hog.compute(im)
        inp_pos_train.append(h)
        print i,len(h)
        out_pos_train.append(1)
    return inp_pos_train, out_pos_train

def train_negative():
    '''
    Training negative elephant sample
    '''
    hog = cv2.HOGDescriptor()
    inp_neg_train = []
    out_neg_train = []
    for i in xrange(1,4):
        if i<10:
            image_no = "/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalNegative/image_000"+str(i)+".jpg"
        else:
            image_no = "/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalNegative/image_00"+str(i)+".jpg"
        im = cv2.imread(image_no)
        h = hog.compute(im)
        inp_neg_train.append(h)
        print i,len(h)
        out_neg_train.append(0)
    return inp_neg_train, out_neg_train
    
def SVM_train(x_train, y_train):
    clf = svm.SVC(C = 5., gamma =0.001)
    clf.fit(x_train, y_train)
    return clf

if __name__ == "__main__":
    x_pos_train, y_pos_train = train_positive()
    x_neg_train, y_neg_train = train_negative()
    main_classifier = SVM_train(x_train, y_train)

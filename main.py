import cv2, numpy
from sklearn import svm

def train_positive():
    '''
    Training positive elephant sample
    '''
    hog = cv2.HOGDescriptor()
    pos_train = []
    for i in xrange(1,79):
        if i<10:
            image_no = "/home/tusharmakkar08/Desktop/ImageProcessing/Data/Total/image_000"+str(i)+".jpg"
        else:
            image_no = "/home/tusharmakkar08/Desktop/ImageProcessing/Data/Total/image_00"+str(i)+".jpg"
        im = cv2.imread(image_no)
        h = hog.compute(im)
        inp_pos_train.append(h)
        out_pos_train.append(1)
    return inp_pos_train, out_pos_train

if __name__ == "__main__":
    print train_positive()

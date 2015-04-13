import cv2, numpy, os
from sklearn import svm

def trainSample(path, totalNo):
    '''
    Training positive elephant sample
    '''
    hog = cv2.HOGDescriptor()
    inp_train = []
    out_train = []
    for i in xrange(1,totalNo):
        if i<10:
            image_no = path + "imagemod_000" + str(i) + ".jpg"
        else:
            image_no = path + "imagemod_00" + str(i) + ".jpg"
        im = cv2.imread(image_no)
        h = hog.compute(im)
        inp_train.append(h)
        print i,len(h)
        out_train.append(1)
    return inp_train, out_train

def SVM_train(x_train, y_train):
    clf = svm.SVC(C = 5., gamma =0.001)
    clf.fit(x_train, y_train)
    return clf
    
def resizeImages(path, totalNo, size):
    for i in xrange(1,totalNo):
        if i<10:
            image_no = path + "image_000" + str(i) + ".jpg"
            image_no_mod = path + "imagemod_000" + str(i)+ ".jpg"
        else:
            image_no = path + "image_00" + str(i) + ".jpg"
            image_no_mod = path + "imagemod_00" + str(i)+ ".jpg"
        cmd = "convert "+image_no+" -resize " + size + "! " + image_no_mod
        #~ print cmd
        os.system(cmd)
        
if __name__ == "__main__":
    #~ resizeImages("/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalPositive/",79, "200x128")
    #~ resizeImages("/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalNegative/",5, "200x128")
    x_pos_train, y_pos_train = trainSample("/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalPositive/", 3)
    x_neg_train, y_neg_train = trainSample("/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalNegative/", 3)
    x_train = x_pos_train + x_neg_train
    y_train = y_pos_train + y_neg_train
    main_classifier = SVM_train(x_train, y_train)

'''
Elephant Detection Code using HOG and SVM
Author: Tushar Makkar <tusharmakkar08[at]gmail.com>
'''

import cv2, numpy, os
from sklearn import svm

def trainSample(hog, path, totalNo, classNo):
    '''
    Training SVM sample using HOG features
    INPUT 
    hog : Hog Descriptor ; send as an parameter for redeclaration and optimizing
    path : Default directory path where images are there
    totalNo : Total Number of images 
    classNo : 0 for non-elephant and 1 for elephant
    OUTPUT 
    inp_train : 2D array of HOG Feature Vector 
    out_train : Correct Labels corresponding to each HOG Vector
    '''
    inp_train = []
    out_train = []
    for i in xrange(1,totalNo + 1):
        if i<10:
            image_no = path + "imagemod_000" + str(i) + ".jpg"
        else:
            image_no = path + "imagemod_00" + str(i) + ".jpg"
        im = cv2.imread(image_no)
        h = hog.compute(im)
        inp_train.append(h.ravel())
        #~ print i,len(h)8
        out_train.append(classNo)
    return inp_train, out_train

def testImage(hog, path, clf):
    '''
    Testing Image by predicting via SVM
    INPUT 
    hog : Hog Descriptor ; send as an parameter for redeclaration and optimizing
    path : Default file path (not directory's)
    clf : SVM Classifier
    OUTPUT 
    Prints whether elephant or not
    '''
    im = cv2.imread(path)
    h = hog.compute(im).ravel()
    #~ print clf.predict(h)
    if clf.predict(h)[0] == 1 : 
        print "Elephant Detected"
    else: 
        print "No Elephant Is Detected"

def SVM_train(x_train, y_train):
    '''
    Training of SVM
    INPUT 
    x_train : Training file with feature Vector
    y_train : Labelled Classes 
    OUTPUT
    Trained SVM Classifier 
    '''
    clf = svm.SVC(C = 5., gamma =0.001)
    clf.fit(x_train, y_train)
    return clf
    
def resizeImages(path, totalNo, size):
    '''
    Code for resizing images so as to have constant length of HOG feature
    vector
    INPUT 
    path : Default directory path where images are there
    totalNo : Total Number of images 
    size : the new size of images
    '''
    for i in xrange(1, totalNo + 1):
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
    '''
    Main function where everything is called
    '''
    #~ resizeImages("/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalPositive/", 78, "200x128")
    #~ resizeImages("/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalNegative/", 78, "200x128")
    hog = cv2.HOGDescriptor()
    x_pos_train, y_pos_train = trainSample(hog, "/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalPositive/", 78, 1)
    x_neg_train, y_neg_train = trainSample(hog, "/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalNegative/", 78, 0)
    x_train = x_pos_train + x_neg_train
    y_train = y_pos_train + y_neg_train
    main_classifier = SVM_train(x_train, y_train)
    testImage(hog, "/home/tusharmakkar08/Desktop/ImageProcessing/Data/TotalNegative/imagemod_0005.jpg", main_classifier)

# elephant_detection
Elephants Detection and Classifier
#### Used : 
* HOG based Features
* SVM Training Model

#### TODO : 
* Crossfold Validation

#### Requirements :
* Python
* OpenCV (cv2)
* numpy
* sklearn

#### How to Run ? 
* Set up the path for positive images, negative images and test images in `main.py` code.
* Use `resizeImages` function to resize them to `200x128`(can fix any size such that the given size <= minimum size of the images). This is done to make the HOG feature vector of same length.
* Make the classifier and save it for future use
* Run `python main.py`

#### How to Test Images ? 
* Do same steps as How to run ? 
* Call `testImage` fucntion with the image name and classifier.

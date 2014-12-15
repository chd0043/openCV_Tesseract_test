
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>

#include <iostream>
#include <math.h>
#include <string.h>
#include <time.h>

#include "DetectLabel.cpp"
#include "LabelOCR.cpp"

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return (-1);

    Mat normalImage, modImage, cropImage1, labelImage1;
    Mat cropImage2, labelImage2, binImage;
    vector<Point> contour;
    vector<vector<Point> > contours;
    Rect label1ROI;

    string text1, text2;

    DetectLabel detectLabels;
    LabelOCR labelOcr;
    CvSVM svmClassifier;

    vector<Mat> possible_labels, label_1, label_2;
    vector<string> labelText1, labelText2;
    detectLabels.showBasicImages = true;
    detectLabels.showAllImages = true;

	namedWindow("normal",WINDOW_NORMAL);

	// SVM learning algorithm

	clock_t begin_time = clock();
	// Read file storage.
	FileStorage fs;
	fs.open("/home/turtlebot/catkin_ws/src/opencv_01/src/vision/ml/SVM.xml", FileStorage::READ);
	Mat SVM_TrainingData;
	Mat SVM_Classes;
	fs["TrainingData"] >> SVM_TrainingData;
	fs["classes"] >> SVM_Classes;
	//Set SVM params
	CvSVMParams SVM_params;
	SVM_params.svm_type = CvSVM::C_SVC;
	SVM_params.kernel_type = CvSVM::LINEAR; //CvSVM::LINEAR;
	SVM_params.degree = 0;
	SVM_params.gamma = 1;
	SVM_params.coef0 = 0;
	SVM_params.C = 1;
	SVM_params.nu = 0;
	SVM_params.p = 0;
	SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
	//Train SVM
	svmClassifier.train( SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params);
	//svmClassifier.train_auto( SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params, 10);

	float timer = ( clock () - begin_time ) /  CLOCKS_PER_SEC;
	cout << "Time: " << timer << endl;

	while(true){

		cap >> normalImage; // get a new frame from camera
		imshow("normal", normalImage);

		possible_labels.clear();
		label_1.clear();
		label_2.clear();

		// segmentation
		detectLabels.segment(normalImage,possible_labels);

		int posLabels = possible_labels.size();
		if (posLabels > 0){
			//For each possible label, classify with svm if it's a label or no
			for(int i=0; i< posLabels; i++)
				{
				if (!possible_labels[i].empty() ){
					Mat gray;
					cvtColor(possible_labels[i], gray, COLOR_RGB2GRAY);
					Mat p= gray.reshape(1, 1);
					p.convertTo(p, CV_32FC1); // CV_32FC1
					int response = (int)svmClassifier.predict( p );
					cout << "Class: " << response << endl;
					if(response==1)
						label_1.push_back(possible_labels[i]);
					if(response==2)
						label_2.push_back(possible_labels[i]);
					}
			}
		}
		if ( label_1.size() > 0) {
			labelText1 = labelOcr.runRecognition(label_1,1);
		}
		if ( label_2.size() > 0) {
			labelText2 = labelOcr.runRecognition(label_2,2);
		}


		if(waitKey(30) >= 0) break;
	}


	return (0);
}

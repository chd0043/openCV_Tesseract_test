/*****************************************************************************

*****************************************************************************/

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>

#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>
#include <sstream>

#include "DetectLabel.cpp"

using namespace std;
using namespace cv;

const string pathLabels1 = "/home/turtlebot/catkin_ws/src/opencv_01/src/vision/ml/LabelDataset/label1_";
const string pathLabels2 = "/home/turtlebot/catkin_ws/src/opencv_01/src/vision/ml/LabelDataset/label2_";
const string path_NoLabels = "/home/turtlebot/catkin_ws/src/opencv_01/src/vision/ml/LabelDataset/noLabel_";

int numLabel1=100;
int numLabel2=100;
int numNoLabels=200;
int imageWidth=400;
int imageHeight=200;

void generateLabelDataset(VideoCapture cap, int numData, int nClass){
	DetectLabel detectLabels;
	detectLabels.showBasicImages = true;
	vector<Mat> label;
	Mat normalImage;
	string path_data;
	int i = 0;


	if (nClass == 1){
	    path_data = pathLabels1;
		numLabel1 = numData;
		}
	else if (nClass == 2){
		path_data = pathLabels2;
		numLabel2 = numData;
	}
	else{
		path_data = path_NoLabels;
		numNoLabels = numData;
	}

	while (i < numData) {
		cap >> normalImage;
		detectLabels.segment(normalImage, label);
		// TODO: RGB to gray
		if (label.size() > 0){
			if (!label[0].empty()){
				stringstream ss;
				ss << path_data << i << ".jpg";
				imwrite(ss.str(), label[0]);
				cout << "path = "<< ss.str() << endl;
				//cout <<  label[0].cols << endl;
				i++;
			}
		}
		namedWindow("normalImage",WINDOW_NORMAL);
		imshow("normalImage",normalImage);
		label.clear();
		if(waitKey(30) >= 0) break;
	}
}

void labelToXml(){
    Mat classes;//(numLabel1+numLabel2+numNoLabels, 1, CV_32FC1);
    Mat trainingData;//(numLabel+numNoLabels, imageWidth*imageHeight, CV_32FC1 );

    Mat trainingImages;
    vector<int> trainingLabels;

    cout << numLabel1 << endl;
    cout << path_NoLabels << endl;

    for(int i=0; i< numLabel1; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << pathLabels1 << i << ".jpg";
        cout << "read path = "<< ss.str() << endl;
        Mat img=imread(ss.str(), 0);
        img= img.reshape(1, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(1);
    }

    for(int i=0; i< numLabel2; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << pathLabels2 << i << ".jpg";
        cout << "read path 2= "<< ss.str() << endl;
        Mat img=imread(ss.str(), 0);
        img= img.reshape(1, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(2);
    }

    for(int i=0; i< numNoLabels; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << path_NoLabels << i << ".jpg";
        cout << "read path 3= "<< ss.str() << endl;
        Mat img=imread(ss.str(), 0);
        if (img.empty()) break;
        img= img.reshape(1, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(0);

    }

    Mat(trainingImages).copyTo(trainingData);
    //trainingData = trainingData.reshape(1,trainingData.rows);
    trainingData.convertTo(trainingData, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);

    FileStorage fs("/home/turtlebot/catkin_ws/src/opencv_01/src/vision/ml/SVM.xml", FileStorage::WRITE);
    fs << "TrainingData" << trainingData;
    fs << "classes" << classes;
    fs.release();
}


int main ( int argc, char** argv )
{
    VideoCapture cap(1); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    int opcion;

    //detectLabels.segment(normalImage,label);

    cout << "OpenCV Training SVM \n";
    cout << "\n";

	while(true){

		//
		cout << "Seleccione que desea hacer: " << endl;
		cout << "1) Tomar muestras positivas de etiqueta 1. " << endl;
		cout << "2) Tomar muestras positivas de etiqueta 2. " << endl;
		cout << "3) Tomar muestras negativas para dataset. " << endl;
		cout << "4) Guardar informacion en XML. " << endl;
		cin >> opcion;

		switch (opcion){
		case 1:
			generateLabelDataset(cap, numLabel1, 1);
			break;
		case 2:
			generateLabelDataset(cap, numLabel2, 2);
			break;
		case 3:
			generateLabelDataset(cap, numNoLabels, 0);
			break;
		case 4:
			labelToXml();
			break;
		default:
			break;
			//


		}



		if(waitKey(30) >= 0) break;
	}


    return 0;
}

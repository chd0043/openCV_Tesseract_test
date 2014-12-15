/*
 * LabelOCR.h
 *
 *  Created on: May 1, 2014
 *      Author: chd
 */

#ifndef LABELOCR_H_
#define LABELOCR_H_

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>

#include <iostream>
#include <math.h>
#include <string.h>
#include <sstream>

#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>

using namespace cv;
using namespace std;

class LabelOCR {

public:
    LabelOCR();
    virtual ~LabelOCR();
    vector<string> runRecognition(const vector<Mat> &labelImage, int labelType);
    tesseract::TessBaseAPI tess;
    bool showImages;

private:
    void preProcess(const Mat &InputImage, Mat &binImage);
    string runPrediction1(const Mat &labelImage, int i);
    string runPrediction2(const Mat &labelImage, int i);
    void skeletonize(Mat& im);
    void thinningIteration(Mat& im, int iter);
    void filterUndesiredChars(string &str);

};

#endif /* LABELOCR_H_ */

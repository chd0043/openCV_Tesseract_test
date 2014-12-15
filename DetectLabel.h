/*
 * DetectLabel.h
 *
 *  Created on: May 1, 2014
 *      Author: chd
 */

#ifndef DETECTLABEL_H_
#define DETECTLABEL_H_

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>

#include <iostream>
#include <math.h>
#include <string.h>
#include <sstream>

using namespace cv;
using namespace std;

const double PI = 3.14159265359;

struct LabelRegion {
    Mat labelImage;
    Mat cropImage;
    string text;
};

class DetectLabel {

public:
    DetectLabel();
    virtual ~DetectLabel();

    void binariza(const Mat &InputImage, Mat &binImage);
    void findRect(const Mat &binImage, vector<vector<Point> > &mark);
    void createLabelMat(const Mat &normalImage, vector<Point> &contour, Mat &labelImage);
    void cropLabelImage(const Mat &normalImage, vector<Point> &contour, Mat &cropImage);
    bool verifySize(vector<Point> &contour);
    void runDetection();
    void segment(const Mat &InputImage, vector<Mat> &output);
    //
    bool showBasicImages;
    bool showAllImages;

private:
    double angle( Point pt1, Point pt2, Point pt0 );
    Point getCenter( vector<Point> points );
    float distanceBetweenPoints( Point p1, Point p2 );
    vector<Point> sortCorners( vector<Point> square );
    void cropImageWithMask(const Mat &img_orig, const Mat &mask, Mat &crop);
    void cropImageColor(const Mat &img, const Mat &cropImage, Mat & color_crop);
    Scalar regionAvgColor(const Mat &img, const Mat &mask);
    bool regionIsCloseToWhite(const Mat &img, const Mat &mask);
    vector<Point> setReducedSquareContour( vector<Point> points );

    //
    Mat blankImage;
    vector<vector<Point> > segments;
    int MaxNumLabels;
    int labelCounter;

};

#endif /* DETECTLABEL_H_ */

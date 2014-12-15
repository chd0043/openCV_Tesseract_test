/*
 * DetectLabel.cpp
 *
 *  Created on: May 1, 2014
 *      Author: chd
 */

#include "DetectLabel.h"

DetectLabel::DetectLabel() {
    //constructor
    int blankWidth  = 640; //
    int blankHeight = 480; //

    blankImage = Mat::ones(blankHeight, blankWidth, CV_8UC3);
    blankImage.setTo(Scalar(255,255,255));

    showBasicImages = true;
    showAllImages = false;
    MaxNumLabels = 2;
    labelCounter = 0;
}

DetectLabel::~DetectLabel() {
    // Auto-generated destructor stub
}

void DetectLabel::binariza(const Mat &InputImage, Mat &binImage)
{
    Mat midImage;

    cvtColor(InputImage, midImage, COLOR_RGB2GRAY);
    threshold(midImage, binImage ,170, 255, CV_THRESH_BINARY);
    //morphologyEx( binImage,binImage,MORPH_CLOSE, Morph);
    if (showAllImages) {
        namedWindow("binImage",WINDOW_NORMAL);
        imshow("binImage",binImage);
    }
}

double DetectLabel::angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return ( (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10) );
}

void DetectLabel::findRect(const Mat& inputImage, vector<vector<Point> > &mark)
{
    Mat edgesImage, binImage;
    vector<vector<Point> > contours;
    vector<Point> obj;
    double maxCosine = 0;

    Canny(inputImage, edgesImage, 10, 20, 3);
    dilate(edgesImage, edgesImage, Mat(), Point(-1,-1));

    findContours(edgesImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    for( size_t i = 0; i < contours.size(); i++ )
    {
        double objLen = arcLength(Mat(contours[i]), true);
        approxPolyDP(Mat(contours[i]), obj, objLen*0.02, true);
        if( obj.size() == 4 && fabs(contourArea(Mat(obj))) > 1000 && isContourConvex(Mat(obj)) )
            mark.push_back(obj);
        //for (int j = 2; j < 5; j++)
        //  {
        //          double cosine = fabs(angle(obj[j%4], obj[j-2], obj[j-1]));
        //          maxCosine = MAX(maxCosine, cosine);
        //  }
        //  if (maxCosine < 0.3)
        //	 mark.push_back(obj);
    }
}

Point DetectLabel::getCenter( vector<Point> points ) {
    Point center = Point( 0.0, 0.0 );

    for( size_t i = 0; i < points.size(); i++ ) {
        center.x += points[ i ].x;
        center.y += points[ i ].y;
    }
    center.x = center.x / points.size();
    center.y = center.y / points.size();
    return (center);
}

float DetectLabel::distanceBetweenPoints( Point p1, Point p2 ) {

    if( p1.x == p2.x ) {
        return ( abs( p2.y - p1.y ) );
    }
    else if( p1.y == p2.y ) {
        return ( abs( p2.x - p1.x ) );
    }
    else {
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        return ( sqrt( (dx*dx)+(dy*dy) ) );
    }
}

vector<Point> DetectLabel::sortCorners( vector<Point> square ) {
    // 0----1
    // |    |
    // |    |
    // 3----2
    Point center = getCenter( square );

    vector<Point> sorted_square;
    for( size_t i = 0; i < square.size(); i++ ) {
        if ( (square[i].x - center.x) < 0 && (square[i].y - center.y) < 0 ) {
            switch( i ) {
            case 0:
                sorted_square = square;
                break;
            case 1:
                sorted_square.push_back( square[1] );
                sorted_square.push_back( square[2] );
                sorted_square.push_back( square[3] );
                sorted_square.push_back( square[0] );
                break;
            case 2:
                sorted_square.push_back( square[2] );
                sorted_square.push_back( square[3] );
                sorted_square.push_back( square[0] );
                sorted_square.push_back( square[1] );
                break;
            case 3:
                sorted_square.push_back( square[3] );
                sorted_square.push_back( square[0] );
                sorted_square.push_back( square[1] );
                sorted_square.push_back( square[2] );
                break;
            }
            break;
        }
    }
    return (sorted_square);
}

void DetectLabel::cropImageWithMask(const Mat &img_orig, const Mat &mask, Mat &crop){
    Mat Image, maskModImage;
    cvtColor(img_orig, Image, CV_BGR2GRAY);
    cvtColor(mask, maskModImage, CV_BGR2GRAY);
    bitwise_and(maskModImage,Image,crop);
}

void DetectLabel::cropImageColor(const Mat &img, const Mat &cropImage, Mat & color_crop){
    Mat ImgCropRGB;
    cvtColor(cropImage, ImgCropRGB, CV_GRAY2RGB);
    bitwise_and(img, ImgCropRGB, color_crop);
}

Scalar DetectLabel::regionAvgColor(const Mat &img, const Mat &mask){
    Scalar avg_color;
    avg_color = mean(img, mask);
    return (avg_color);
}

void DetectLabel::createLabelMat(const Mat &normalImage, vector<Point> &contour, Mat &labelImage){

    labelImage = Mat::zeros(200, 400, CV_8UC3);
    Point2f labelPoints[4], pPerspOrig[4], contourPersp[4];
    float distanceP0P1, distanceP0P2, distanceP0P3;

    //contour = sortCorners(contour);
    contourPersp[0] = contour[0];
    contourPersp[1] = contour[1];
    contourPersp[2] = contour[2];
    contourPersp[3] = contour[3];

    distanceP0P1 = distanceBetweenPoints( contourPersp[0], contourPersp[1] );
    distanceP0P2 = distanceBetweenPoints( contourPersp[0], contourPersp[2] );
    distanceP0P3 = distanceBetweenPoints( contourPersp[0], contourPersp[3] );

    if ( distanceP0P1 < distanceP0P3 ) {
        labelPoints[0]=(Point2f(0, 0));
        labelPoints[1]=(Point2f(0, labelImage.rows));
        labelPoints[2]=(Point2f(labelImage.cols, labelImage.rows));
        labelPoints[3]=(Point2f(labelImage.cols, 0));
    }
    else {
        labelPoints[1]=(Point2f(0, 0));
        labelPoints[2]=(Point2f(0, labelImage.rows));
        labelPoints[3]=(Point2f(labelImage.cols, labelImage.rows));
        labelPoints[0]=(Point2f(labelImage.cols, 0));
    }

    Mat transmtx = getPerspectiveTransform(contourPersp, labelPoints);
    warpPerspective(normalImage, labelImage, transmtx, labelImage.size());
}

void DetectLabel::cropLabelImage(const Mat &normalImage, vector<Point> &contour, Mat &cropImage){
    // declare used vars
    Point2f contourPersp[4], pPerspOrig[4];
    Mat arMask;

    contour = sortCorners(contour);
    //
    contourPersp[0] = contour[0];
    contourPersp[1] = contour[1];
    contourPersp[2] = contour[2];
    contourPersp[3] = contour[3];

    pPerspOrig[0] = Point( 0, 0 );
    pPerspOrig[1] = Point( blankImage.cols, 0 );
    pPerspOrig[2] = Point( blankImage.cols, blankImage.rows );
    pPerspOrig[3] = Point( 0, blankImage.rows);

    Mat warpMatrix = getPerspectiveTransform(pPerspOrig, contourPersp);
    warpPerspective(blankImage, arMask, warpMatrix, blankImage.size());//, INTER_LINEAR, BORDER_CONSTANT );
    //imshow("arMask",arMask);

    cropImageWithMask(normalImage, arMask, cropImage);
}

bool DetectLabel::regionIsCloseToWhite(const Mat &img, const Mat &mask){
    float rChannelMax = 150;
    float gChannelMax = 150;
    float bChannelMax = 150;
    Scalar avg_color;

    avg_color = mean(img);

    cout << "color: ";
    cout << avg_color << endl;

    return (avg_color[0] > bChannelMax &&
            avg_color[1] > bChannelMax &&
            avg_color[2] > bChannelMax );
}


vector<Point> DetectLabel::setReducedSquareContour( vector<Point> points ) {

    vector<Point> modPoints;
    // 1----2
    // |    |
    // |    |
    // 0----3
    modPoints = points;
    if (points.size() == 4) {
        modPoints[0].x = (points[1].x + points[0].x) / 2;
        modPoints[0].y = (points[1].y + points[0].y) / 2;
        modPoints[1] = points[1];
        modPoints[3].x = (points[2].x + points[3].x) / 2;
        modPoints[3].y = (points[2].y + points[3].y) / 2;
        modPoints[2] = points[2];
    }
    return (modPoints);
}//int labelCounter;

bool DetectLabel::verifySize(vector<Point> &contour){

    float MaxPerimeter = 1450;
    float MinPerimeter = 530;
    float MaxArea = 165000;
    float MinArea = 12000;

    float perimeter = arcLength(contour,true);
    float area = contourArea(contour);

    return  (area > MinArea && area < MaxArea);

}

void DetectLabel::segment(const Mat &InputImage, vector<Mat> &outputImage){
    //vector<LabelRegion> output;
    Mat binImage;
    vector<Mat> cropImage;
    vector<vector<Point> > auxSegments; // segments
    int MaxIter;

    segments.clear();

    binariza(InputImage, binImage);
    findRect(binImage, segments);

    outputImage.resize(segments.size());
    cropImage.resize(segments.size());

    labelCounter = 0;
    MaxIter = segments.size();

    for( size_t i = 0; i < MaxIter; i++ )
    {
        if (i > MaxNumLabels-1)
            break;
        if( verifySize(segments[i]) ) {
            cropLabelImage(InputImage, segments[i], cropImage[i]);
            createLabelMat(InputImage, segments[i], outputImage[i]);
            auxSegments.push_back(segments[i]);
            labelCounter++;

            // ----- show pictures ------- //
            if (showBasicImages || showAllImages) {
                stringstream ss; ss << i;
                string str = ss.str();
                Mat modImage = InputImage.clone();
                drawContours(modImage, segments, -1, Scalar(0, 255, 0), 3 );
                imshow("normal", modImage);
            }
            if (showAllImages) {
                stringstream ss; ss << i;
                string str = ss.str();
                namedWindow("cropImage_"+str,WINDOW_NORMAL);
                //namedWindow("cropImage_0",WINDOW_NORMAL);
                imshow("cropImage_"+str, cropImage[i]);
                //imshow("cropImage_0", cropImage[i]);
            }
        }
    }

}

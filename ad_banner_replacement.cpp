#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <queue>
#include <set>
#include <Eigen/Dense>

using namespace cv;
using std::cout;
using std::endl;

struct ValueType{
    std::vector<Vec4i> line;
    double angel;
};

struct Comparator {
    bool operator()(std::tuple<double, Vec4i, double, double>& t1, 
            std::tuple<double, Vec4i, double, double>& t2) {
         return std::get<0>(t1) < std::get<0>(t2);
     }
 };

Mat biggestAreaMask(Mat mask)
{   
    Mat labels, stats, centroids;
    connectedComponentsWithStats(mask, labels, stats, centroids);

    int maxRow = -1, maxSize = 0;
    for (int row = 1; row < stats.rows; row++){
        if (stats.row(row).at<int>(4) > maxSize){
            maxRow = row;
            maxSize = stats.row(row).at<int>(4);
        }
    }
    mask = labels == maxRow;
    return mask;
}

Mat pitchMask(Mat img)
{
    Mat imgHSV, mask;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);
    
    Scalar lowerBound = Scalar(36, 50, 70), upperBound = Scalar(85, 255, 255);
    inRange(imgHSV, lowerBound, upperBound, mask);

    int morph_size = 3;
    Mat kernel = getStructuringElement( MORPH_RECT,
                       Size( 2*morph_size + 1, 2*morph_size+1 ),
                       Point( morph_size, morph_size ) );

    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1,1), 2);

    mask = biggestAreaMask(mask);

    return mask;
}

std::vector<Point> findCorners(std::vector<Point> points)
{
    Point topLeft = points[0];
    Point topRight = points[0];
    Point bottomLeft = points[0];
    Point bottomRight = points[0];

    for (int i = 1; i < points.size(); i++){
        int x = points[i].x;
        int y = points[i].y;
        if (x*x + y*y < topLeft.x*topLeft.x + topLeft.y*topLeft.y){
            topLeft = points[i];
        }
        else if (x + y > bottomRight.x + bottomRight.y){
            bottomRight = points[i];
        }
        else if (x - y > topRight.x - topRight.y){
            topRight = points[i];
        }
        else if (y - x > bottomLeft.y - bottomLeft.x){
            bottomLeft = points[i];
        }
    }

    std::vector<Point> corners {topLeft, topRight, bottomRight, bottomLeft};
    return corners;
}

Mat outLimitBannerMask(Mat innerLimitMask)
{   
    Mat outerLimitMask = Mat::zeros(innerLimitMask.rows, innerLimitMask.cols, CV_64FC1);

    int heightTolerance = (int)innerLimitMask.rows * 0.025;
    int height = (int)innerLimitMask.rows * 0.09;
    int widthTolerance = (int)innerLimitMask.cols * 0.005;
    int width = (int)innerLimitMask.cols * 0.04;

    for (int row = 0; row < innerLimitMask.rows; row++){
        for (int col = 0; col < innerLimitMask.cols; col++){

            // Introduce height tolerance to mask
            if (row >= heightTolerance 
            && innerLimitMask.at<double>(row, col) == 1 
            && innerLimitMask.at<double>(row - heightTolerance, col) == 0) {
                outerLimitMask.at<double>(row, col) = 1;
            }

            // Introduce outer limit to mask
            else if (row < innerLimitMask.rows - height && innerLimitMask.at<double>(row, col) == 0 && innerLimitMask.at<double>(row + height, col) == 1) {
                outerLimitMask.at<double>(row, col) = 1;
            }
        }
    }

    return outerLimitMask;
}

Mat bannerMask(Mat pitchMask)
{
    Mat pitchMaskPadded;
    int padding = 10;
    copyMakeBorder(pitchMask, pitchMaskPadded, padding, padding, padding, padding, BORDER_CONSTANT);

    std::vector<Point> points;
    goodFeaturesToTrack(pitchMaskPadded, points, 100, 0.01, 20);

    std::vector<Point> corners = findCorners(points);

    Mat maskBGRPadded;
    cvtColor(pitchMaskPadded, maskBGRPadded, COLOR_GRAY2BGR);

    for (size_t r = 0; r < corners.size(); r++)
    {   
        circle(maskBGRPadded, corners[r], 5, Scalar(0,0,255), -1);
    }

    Mat maskPadded = Mat::zeros(maskBGRPadded.rows, maskBGRPadded.cols, CV_64FC1);
    fillConvexPoly(maskPadded, corners, Scalar(1));

    Mat maskUnpadded = cv::Mat(maskPadded, cv::Rect(padding, padding, maskPadded.cols - 2 * padding, 
                            maskPadded.rows - 2 * padding));
    Mat mask = outLimitBannerMask(maskUnpadded);
    mask.convertTo(mask, CV_8U);
    mask = biggestAreaMask(mask);

    return mask;
}

Mat cannyEdgeDetection(
    Mat img, 
    const int lowThreshold = 100,
    const int highThreshold = 200,
    const int kernelSize = 3)
{   
    Mat greenCh;
    extractChannel(img, greenCh, 1);
    Mat greenMask = greenCh > 250;

    Mat imgGray, imgGaussianBlur33, imgCannyG33;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    GaussianBlur(imgGray, imgGaussianBlur33, Size(3,3), 1.5);
    Canny(imgGaussianBlur33, imgCannyG33, lowThreshold, highThreshold, kernelSize);

    return imgCannyG33;
}

std::vector<Vec4i> houghLines(Mat img)
{
    Mat imgBGR;
    std::vector<Vec4i> houghLines;

    cvtColor(img, imgBGR, COLOR_GRAY2BGR);

    HoughLinesP(img, houghLines, 1, CV_PI/720, 200, 0, 400);

    int longest1 = 0;
    int longest2 = 0;
    int idx1, idx2;
    for (size_t i = 0; i < houghLines.size(); i++) {
        Vec4i l = houghLines[i];
        double len = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
        if (len > longest1) {
            longest2 = longest1;
            longest1 = len;

            idx2 = idx1;
            idx1 = i;
        }
        else if (len > longest2) {
            longest2 = len;
            idx2 = i;
        }
        line(imgBGR, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, LINE_AA);
    }
    
    return houghLines;
}

bool compareWithRange(double val, std::set<double> compVals, double tolerance)
{
    for (double compVal : compVals){
        if (val >= compVal - tolerance && val <= compVal + tolerance){
            return true;
        }
    }
    return false;
}

std::vector<Vec4i> longestTwo(std::vector<Vec4i> lines, int X=2, double angleTol=5.0, double cTol=10)
{
    std::vector<std::tuple<double, Vec4i, double, double>> infoTuples;
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        double len = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
        double angle = atan((l[3] - l[1]) / (double)(l[2] - l[0])) * 360 / CV_PI;
        double c = (l[1]*l[2] - l[0]*l[3]) / double(l[2] - l[0] + 0.000001);  // from line equation y = mx + c
        infoTuples.push_back(std::tuple<double, Vec4i, double, double>(len, l, angle, c));
    }

    std::priority_queue<std::tuple<double, Vec4i, double, double>, 
        std::vector<std::tuple<double, Vec4i, double, double>>, Comparator> infoQueue;

    for(const auto& elem : infoTuples){
        infoQueue.push(elem);
    }

    std::vector<Vec4i> longestTwoLines = {std::get<1>(infoQueue.top())};
    std::set<double> seenAngles, seenCs;
    seenAngles.insert(std::get<2>(infoQueue.top()));
    seenCs.insert(std::get<3>(infoQueue.top()));
    infoQueue.pop();

    int count = 1;
    while (count < X && !infoQueue.empty()) {
        std::tuple<double, Vec4i, double, double> candidate = infoQueue.top();
        Vec4i candidateLine = std::get<1>(candidate);
        double candidateAngle = std::get<2>(candidate);
        double candidateC = std::get<3>(candidate);

        bool checkAngle = compareWithRange(candidateAngle, seenAngles, angleTol);
        bool checkC = compareWithRange(candidateC, seenCs, cTol);

        if (!checkAngle || !checkC){
            longestTwoLines.push_back(candidateLine);
            count++;
        }
        infoQueue.pop();
    }

    cout << "Longest 2 lines: " << longestTwoLines[0] << " " << longestTwoLines[1] << "\n";

    return longestTwoLines;
}

bool CustomVectorCompare(const Vec4i &first, const Vec4i &second)
{      
    return first[1] < second[1];
}

std::vector<Point> extendLines(std::vector<Vec4i> lines, Mat img)
{   
    // sort lines in increasing y-coordinate to ensure top left & right corners are added first;
    std::sort(lines.begin(), lines.end(), CustomVectorCompare);

    int width = img.cols;
    int height = img.rows;

    Eigen::Vector3d top_left(0, 0, 1);
    Eigen::Vector3d top_right(width-1, 0, 1);
    Eigen::Vector3d bottom_left(0, height-1, 1);
    Eigen::Vector3d bottom_right(width-1, height-1, 1);

    Eigen::Vector3d left_border = top_left.cross(bottom_left);
    Eigen::Vector3d left_border2 = bottom_left.cross(top_left);
    Eigen::Vector3d right_border = top_right.cross(bottom_right);

    std::vector<Point> pts;
    for (Vec4i line : lines){
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];

        Eigen::Vector3d v1(x1, y1, 1);
        Eigen::Vector3d v2(x2, y2, 1);

        Eigen::Vector3d l = v1.cross(v2);

        Eigen::Vector3d p1 = l.cross(left_border);
        p1 = p1 / p1[2];
        Eigen::Vector3d p2 = l.cross(right_border);
        p2 = p2 / p2[2];

        pts.push_back(Point(p1[0], p1[1]));
        pts.push_back(Point(p2[0], p2[1]));
    }

    return pts;
}

std::vector<Point> uncrop(std::vector<Point> pts, int xMin, int yMin)
{
    std::vector<Point> uncroppedPts;
    for (Point pt : pts){
        Point uncroppedPt = pt + Point(xMin, yMin);
        uncroppedPts.push_back(uncroppedPt);
    }
    return uncroppedPts;
}

std::vector<int> maskCorners(Mat mask){
    int xMin = mask.cols - 1;
    int xMax = 0;
    int yMin = mask.rows - 1;
    int yMax = 0;
    for (int row = 0; row < mask.rows; row ++){
        for (int col = 0; col < mask.cols; col++) {
            if (mask.at<char>(row, col) != 0) {
                if (col < xMin){
                    xMin = col;
                }
                if (col > xMax){
                    xMax = col;
                }
                if (row < yMin){
                    yMin = row;
                }
                if (row > yMax){
                    yMax = row;
                }
            }
        }
    }
    std::vector<int> corners {xMin, xMax, yMin, yMax};
    return corners;
}

int main(int argc, char *argv[])
{
    // Read image from file
    Mat img = imread(argv[1]);
    // if fail to read the image
    if (img.empty())
    {
        cout << "Error loading the image" << endl;
        return -1;
    }

    namedWindow("Original Img", WINDOW_AUTOSIZE);
    imshow("Original Img", img);
    waitKey(0);

    Mat pMask = pitchMask(img);
    Mat bMask = bannerMask(pMask);

    // Read template from file
    Mat templ = imread(argv[2]);
    templ = repeat(templ, 1, 10);
    // if fail to read the template
    if (templ.empty())
    {
        cout << "Error loading the template" << endl;
        return -1;
    }
    Mat maskedImg;

    // Crop to mask
    std::vector<int> corners = maskCorners(bMask);
    int xMin = corners[0];
    int xMax = corners[1];
    int yMin = corners[2];
    int yMax = corners[3];
    Rect cropRegion(xMin, yMin, xMax - xMin, yMax - yMin);
    maskedImg = img(cropRegion);

    Mat imgCanny;
    std::vector<Vec4i> lines, longestLines;

    imgCanny = cannyEdgeDetection(maskedImg);
    lines = houghLines(imgCanny);
    longestLines = longestTwo(lines);

    Mat imgBGR;
    cvtColor(imgCanny, imgBGR, COLOR_GRAY2BGR);
    line(imgBGR, Point(longestLines[0][0], longestLines[0][1]), 
        Point(longestLines[0][2], longestLines[0][3]), 
        Scalar(255, 0, 0), 3, LINE_AA);
    line(imgBGR, Point(longestLines[1][0], longestLines[1][1]), 
        Point(longestLines[1][2], longestLines[1][3]), 
        Scalar(255, 0, 0), 3, LINE_AA);

    std::vector<Point> pts = extendLines(longestLines, imgCanny);
    pts = uncrop(pts, xMin, yMin);

    int bannerHeight = ((pts[2].y - pts[0].y) + (pts[3].y - pts[1].y)) / 2;
    int bannerWidth = ((pts[1].x - pts[0].x) + (pts[3].x - pts[2].x)) / 2;

    cout << "banner height=" << bannerHeight << " banner width=" << bannerWidth << endl;

    double imgHeightToWidth = (double)bannerHeight / (double)bannerWidth;
    double templateHeightToWidth = double(templ.rows) / double(templ.cols);

    Rect crop;
    crop.x = 0;
    crop.y = 0;
    
    if (imgHeightToWidth > templateHeightToWidth)
    {
        crop.height = templ.rows;
        crop.width = templ.rows / imgHeightToWidth;
    }
    else
    {
        crop.height = templ.cols * imgHeightToWidth;
        crop.width = templ.cols;
    }

    Mat templCrop = templ(crop);

    std::vector<Point> templPts {
        Point(0, 0), 
        Point(templCrop.cols, 0), 
        Point(0, templCrop.rows),
        Point(templCrop.cols, templCrop.rows)};

    std::vector<Point2f> pts2f;
    for (int i = 0; i < pts.size(); i++){
        pts2f.push_back((Point2d)pts[i]);
    }
    std::vector<Point2f> templPts2f;
    for (int i = 0; i < templPts.size(); i++){
        templPts2f.push_back((Point2d)templPts[i]);
    }

    // replace banner advertisement with template
    Mat H = getPerspectiveTransform(templPts2f, pts2f);
    cv::Mat composite;
    img.copyTo(composite);
    warpPerspective(templCrop, composite, H, composite.size(), cv::INTER_CUBIC,cv::BORDER_TRANSPARENT);

    imshow("warped template", composite);
    waitKey(0);

    String outPath = argv[1];
    int length = outPath.length();
    outPath.erase(length - 5, 5);
    outPath += "_result.jpeg";

    imwrite(outPath, composite);

    return 0;
}
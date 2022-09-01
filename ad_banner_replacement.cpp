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

struct Comparator 
{
    bool operator()(std::tuple<double, Vec4i, double, double> t1, 
            std::tuple<double, Vec4i, double, double> t2) {
                return std::get<0>(t1) < std::get<0>(t2);
                }
};

Mat getTemplate(String templatePath)
{
    // Read template from file
    Mat templ = imread(templatePath);
    templ = repeat(templ, 1, 10);
    // if fail to read the template
    if (templ.empty())
    {
        cout << "Error loading the template" << endl;
    }
    return templ;
}

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

std::vector<Point> findCorners(Mat pMaskPadded)
{
    std::vector<Point> points;
    goodFeaturesToTrack(pMaskPadded, points, 100, 0.01, 20);

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

Mat bannerMask(Mat img)
{   
    // detect the pitch on the image.
    Mat pMask = pitchMask(img);

    // pad the pitch mask to also detect the image corners as potential corners of the pitch
    Mat pMaskPadded;
    int padding = 10;
    copyMakeBorder(pMask, pMaskPadded, padding, padding, padding, padding, BORDER_CONSTANT);

    // find the corners of the pitch
    std::vector<Point> corners = findCorners(pMaskPadded);

    // create a mask spanning the 4 pitch corners and unpad
    Mat maskPadded = Mat::zeros(pMaskPadded.rows, pMaskPadded.cols, CV_64FC1);
    fillConvexPoly(maskPadded, corners, Scalar(1));
    Mat maskUnpadded = cv::Mat(maskPadded, cv::Rect(padding, padding, maskPadded.cols - 2 * padding, 
                            maskPadded.rows - 2 * padding));

    // create a mask for the banners surrounding the pitch
    Mat mask = outLimitBannerMask(maskUnpadded);
    mask.convertTo(mask, CV_8U);
    mask = biggestAreaMask(mask);

    return mask;
}

void getMaskCorners(Mat mask, int &xMin, int &xMax, int &yMin, int &yMax){
    xMin = mask.cols - 1;
    xMax = 0;
    yMin = mask.rows - 1;
    yMax = 0;
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

    for(std::tuple<double, Vec4i, double, double> elem : infoTuples){
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

bool CustomVectorCompare(Vec4i first, const Vec4i second)
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

void uncrop(std::vector<Point> &pts, int xMin, int yMin)
{
    std::vector<Point> uncroppedPts;
    for (Point pt : pts){
        Point uncroppedPt = pt + Point(xMin, yMin);
        uncroppedPts.push_back(uncroppedPt);
    }
    pts = uncroppedPts;
}

void replaceAdBanner(Mat img, String imgPath, String templPath)
{
    namedWindow("Original Img", WINDOW_AUTOSIZE);
    imshow("Original Img", img);
    waitKey(0);

    Mat templ = getTemplate(templPath);

    Mat bMask = bannerMask(img);

    // Crop image to mask area
    int xMin, xMax, yMin, yMax;
    getMaskCorners(bMask, xMin, xMax, yMin, yMax);
    Rect cropRegion(xMin, yMin, xMax - xMin, yMax - yMin);
    Mat maskedImg = img(cropRegion);

    // Detect the banner border lines using Canny Edge Detection and Hough Transform for line detection
    Mat imgCanny = cannyEdgeDetection(maskedImg);
    std::vector<Vec4i> lines = houghLines(imgCanny);
    std::vector<Vec4i> longestLines = longestTwo(lines);

    // get the corner points of the banner, in the coordinate system of the uncropped image
    std::vector<Point> pts = extendLines(longestLines, imgCanny);
    uncrop(pts, xMin, yMin);

    // compute the ratio "height:with" for the banner and template
    int bannerHeight = ((pts[2].y - pts[0].y) + (pts[3].y - pts[1].y)) / 2;
    int bannerWidth = ((pts[1].x - pts[0].x) + (pts[3].x - pts[2].x)) / 2;
    cout << "banner height=" << bannerHeight << " banner width=" << bannerWidth << endl;

    double bannerHeightToWidth = (double)bannerHeight / (double)bannerWidth;
    double templateHeightToWidth = double(templ.rows) / double(templ.cols);

    // crop the template to the same "height:width"-ratio as the banner
    Rect crop;
    crop.x = 0;
    crop.y = 0;
    if (bannerHeightToWidth > templateHeightToWidth){
        crop.height = templ.rows;
        crop.width = templ.rows / bannerHeightToWidth;
    }
    else{
        crop.height = templ.cols * bannerHeightToWidth;
        crop.width = templ.cols;
    }
    Mat templCrop = templ(crop);

    // define the corners of the template
    std::vector<Point2d> templPts {
        Point(0, 0), 
        Point(templCrop.cols, 0), 
        Point(0, templCrop.rows),
        Point(templCrop.cols, templCrop.rows)};

    // cast points to Point2d
    std::vector<Point2f> pts2d;
    for (int i = 0; i < pts.size(); i++){
        pts2d.push_back((Point2d)pts[i]);
    }
    std::vector<Point2f> templPts2d;
    for (int i = 0; i < templPts.size(); i++){
        templPts2d.push_back((Point2d)templPts[i]);
    }

    // compute the homography and replace the banner with the template
    Mat H = getPerspectiveTransform(templPts2d, pts2d);
    cv::Mat composite;
    img.copyTo(composite);
    warpPerspective(templCrop, composite, H, composite.size(), cv::INTER_CUBIC,cv::BORDER_TRANSPARENT);

    imshow("warped template", composite);
    waitKey(0);

    // store the result to the same image path, just adding "_result" to the file name
    String outPath = imgPath;
    int length = outPath.length();
    outPath.erase(length - 5, 5);
    outPath += "_result.jpeg";

    imwrite(outPath, composite);
}

int main(int argc, char *argv[])
{
    String imgPath = argv[1]; 
    String templPath = argv[2];

    // Read image from file
    Mat img = imread(imgPath);
    // if fail to read the image
    if (img.empty())
    {
        cout << "Error loading the image" << endl;
        return -1;
    }

    replaceAdBanner(img, imgPath, templPath);

    return 0;
}
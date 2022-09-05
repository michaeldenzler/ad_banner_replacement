#include <opencv2/opencv.hpp>

bool hasEnding (cv::String fullString, cv::String ending);
cv::Mat getTemplate(cv::String templatePath);
bool customVectorCompare(cv::Vec4i first, cv::Vec4i second);
cv::String createOutPath(cv::String filePath, cv::String suffix);
bool compareWithRange(double val, std::set<double> compVals, double tolerance);
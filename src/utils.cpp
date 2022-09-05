#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * Checks wheter a string has a specific ending.
 *
 * @param fullString The string to check.
 * @param ending The string ending to check for.
 * @return True if the string ends on the specific ending, false otherwise.
 */
bool hasEnding (cv::String fullString, cv::String ending) 
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

/**
 * Load the template from its path.
 *
 * @param templatePath The path to the template file.
 * @return The template.
 */
cv::Mat getTemplate(cv::String templatePath)
{
    // Read template from file
    cv::Mat templ = cv::imread(templatePath);
    templ = cv::repeat(templ, 1, 10);
    // if fail to read the template
    if (templ.empty())
    {
        std::cout << "Error loading the template" << std::endl;
    }
    return templ;
}

/**
 * Comparison function to sort a vector of lines with nondecreasing y1-values.
 *
 * @param first The first line of the comparison.
 * @param second The second line of the comparison.
 * @return Whether the first line segment has a smaller y1-value than the second line segment.
 */
bool customVectorCompare(cv::Vec4i first, cv::Vec4i second)
{      
    return first[1] < second[1];
}

/**
 * Creates the path string of the output file using the same directory as the input file's.
 *
 * @param filePath The path of the input file.
 * @param templPath The format suffix to use for the output file.
 * @return The output path.
 */
cv::String createOutPath(cv::String filePath, cv::String suffix)
{
    // store the result to the same image path, just adding "_result" to the file name
    cv::String outPath = filePath;
    int length = outPath.length();
    outPath.erase(length - suffix.length(), 5);
    outPath += "_result" + suffix;
    return outPath;
}

/**
 * Compares a value with a set of values and some error tolerance.
 *
 * @param val The value to compare.
 * @param compVals A set of values to compare the input value with.
 * @param tolerance The tolerance for error when comparing values.
 * @return Whether the value is close enough to any of the comparison values or not.
 */
bool compareWithRange(double val, std::set<double> compVals, double tolerance)
{
    for (double compVal : compVals){
        if (val >= compVal - tolerance && val <= compVal + tolerance){
            return true;
        }
    }
    return false;
}
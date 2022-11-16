#ifndef _IR_FILTER_H
#define _IR_FILTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
 class IR_filter
 {
  public:
    IR_filter(std::string imgPath):src(cv::imread(imgPath, cv::ImreadModes::IMREAD_GRAYSCALE)){
        if(src.empty()){
            std::cout << "Image not found!" << std::endl;
            exit(1);
        }
    };
    // 高斯拉普拉斯滤波，提取边信息，然后可用作加深边缘信息。
    cv::Mat LaplacianOfGaussianFilter(cv::Mat src, int windowSize, double sigma);
    // 边保留滤波、降噪的同时保留边缘信息。 
    cv::Mat LocalEdgePreservingFilter(cv::Mat src,  float alpha=0.1, float beta=1.0, int windowSize=4, bool isFloat=true);
    // 引导滤波， 降噪的同时增强细节
    cv::Mat GuidedFilter(cv::Mat src, cv::Mat guide, int windowSize, double eps);
    // 增强的双边滤波
    cv::Mat BilateralFilter(cv::Mat src, int windowSize, double sigmaColor, double sigmaSpace);
    // Bi-exponential edge preserving smoother
    cv::Mat BiExponentialFilter(cv::Mat src, float sigma, float lambda);
    cv::Mat src;
  protected:
    

    // The 2 Dimensional Least mean square filter
    cv::Mat LMS_2D(cv::Mat src, int windowSize, double stepSize, int iteration);

    
    
  private:
    float LoGFunction(int x, int y, double sigma);
    float gaussianFunction(float x, float y, float sigma);
    cv::Mat BEEPS(cv::Mat src, float sigma, float lambda, bool isVertical);
 };


#endif
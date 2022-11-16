#ifndef _COWC_H_
#define _COWC_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
class cowc{
    public:
        cowc(const std::string& srcPath):src(cv::imread(srcPath, cv::IMREAD_GRAYSCALE)){
            if(src.empty()){
                std::cout << "Image is empty!" << std::endl;
            }
        }
        ~cowc(){
            std::cout<<"~cowc()"<<std::endl;
        }
        cv::Mat src;
    private:
        // Discrete Wavelet Transform haar decomposition
        cv::Mat DWT(cv::Mat& src, int level);
        
};

#endif
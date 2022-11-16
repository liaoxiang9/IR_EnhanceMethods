#ifndef _ADPHE_H_
#define _ADPHE_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

class adphe{
    public:
        adphe(const std::string& imagePath):src(cv::imread(imagePath, 0)){
            if(src.empty()){
                std::cout << "image is empty" << std::endl;
            }
        }
        void set(const std::string&);
        cv::Mat mainProcess();
    protected:
        cv::Mat src;
        cv::Mat hist;
        const int histSize = 256;
        const int winSize = 3;
        float CV0 = 0.0;
        float CV1 = 0.0;
        void getHist();
        float* getLimit();
        cv::Mat clipHist(const cv::Mat&,float lowerLimit, float upperLimit);
        float getCV(const cv::Mat&);
        
};


#endif
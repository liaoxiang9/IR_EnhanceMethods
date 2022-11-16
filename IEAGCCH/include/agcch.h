#ifndef _AGCCH_H_
#define _AGCCH_H_

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

class agcch{
    public:
        agcch(const std::string& imagePath, const int& winSize): src(cv::imread(imagePath, 0)), winSize(winSize){
            if(src.empty()){
                std::cout << "Image is empty!" << std::endl;
            }
        }

        void setWinSize(int winSize){
            this->winSize = winSize;
        }

        ~agcch();
        cv::Mat iegagc();

        cv::Mat ielagc();
        cv::Mat getSrc(){
            return src;
        }
    protected:
        cv::Mat src;
        cv::Mat hist;
        int winSize = 7;
        int histSize = 256;
        void getHist();
        
};



#endif
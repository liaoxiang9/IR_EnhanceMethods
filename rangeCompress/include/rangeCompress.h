#ifndef _RANGE_COMPRESS_H_
#define _RANGE_COMPRESS_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>

class rangeCompress
{
    public:
    // read 16bit image
        rangeCompress(){
            std::cout<<"rangeCompress"<<std::endl;
        };
        ~rangeCompress(){
            std::cout<<"~rangeCompress"<<std::endl;
        };
        cv::Mat histogramCompress(cv::Mat &src,float threshPercent);
        cv::Mat gammaCurveCompress(cv::Mat &src, float gamma);
        cv::Mat linearCompress(cv::Mat &src);
    protected:
    private:
        unsigned int inputBitWidth=65536;
        unsigned int outputBitWidth=256;
        // get entropy of the image
        float getEntropy(cv::Mat &src);
};


#endif
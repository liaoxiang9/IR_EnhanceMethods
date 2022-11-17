#ifndef _IR_DENOISE_H
#define _IR_DENOISE_H_

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <iostream>


class IR_denoise
{
    public:
        IR_denoise(){
            std::cout<<"IR_denoise constructor"<<std::endl;
        };
        virtual ~IR_denoise(){
            std::cout<<"IR_denoise destructor"<<std::endl;
        };
        void set_image(std::string path){
            this->src = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if(src.empty()){
                std::cout << "Error: Image not found!" << std::endl;
                exit(1);
            }
        };
        virtual void denoise(cv::Mat& dst) const = 0;

    protected:
        cv::Mat src;
        cv::Mat dst;
};
#endif




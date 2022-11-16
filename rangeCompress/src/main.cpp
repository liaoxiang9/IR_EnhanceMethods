#include "rangeCompress.h"
#include <iostream>
#include <string>
int main(){
    rangeCompress rc;
    cv::Mat src = cv::imread("/home/lx/IR_Enhance_Methods/src_images/FLIR_00001.tiff", -1);
    if(src.empty()){
        std::cout<<"The image is empty"<<std::endl;
        return -1;
    }
    cv::Mat compressedImage = rc.histogramCompress(src, 0.0005);
    cv::imwrite("/home/lx/IR_Enhance_Methods/rangeCompress/result/hc.jpg", compressedImage);

    cv::Mat linearCompressedImage = rc.linearCompress(src);
    cv::imwrite("/home/lx/IR_Enhance_Methods/rangeCompress/result/lc.jpg", linearCompressedImage);

    cv::Mat gammaCurveCompressedImage = rc.gammaCurveCompress(src, 1.6);
    // cout the min and max value of the gammaCurveCompressedImage
    // double minVal, maxVal;
    // cv::minMaxLoc(gammaCurveCompressedImage, &minVal, &maxVal);
    // std::cout<<"minVal: "<<minVal<<std::endl;
    // std::cout<<"maxVal: "<<maxVal<<std::endl;
    cv::imwrite("/home/lx/IR_Enhance_Methods/rangeCompress/result/gcc.jpg", gammaCurveCompressedImage);
}
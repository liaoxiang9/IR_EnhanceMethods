#include "contrast_stretch.h"

void contrast_stretch::set_src_img(const std::string &image_path){
    this->src_img = cv::imread(image_path, 0);
}

cv::Mat contrast_stretch::contrast_ratio(){
    double min = 0.0;
    double max = 0.0;
    cv::minMaxLoc(src_img, &min, &max);
    double target_value = 0;
    for(int i=0;i<src_img.rows;i++){
        for(int j=0;j<src_img.cols;j++){
            target_value = ((double)src_img.at<uchar>(i, j) - min) / (max - min) * 255;
            src_img.at<uchar>(i, j) = (uchar)target_value;
        }
    }
    return src_img;
}









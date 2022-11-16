#ifndef CONTRAST_STRETCH_H_
#define CONTRAST_STRETCH_H_
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <iostream>
class contrast_stretch
{
public:
    // 设置读取图片的路径
    void set_src_img(const std::string &);
    cv::Mat contrast_ratio();
    contrast_stretch(const std::string &path): src_img(cv::imread(path, 0)) {}
private:
    cv::Mat src_img;  
};

#endif

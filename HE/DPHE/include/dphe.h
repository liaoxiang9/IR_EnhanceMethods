#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>
#include <string>
#include "/home/lx/IR_Enhance_Methods/HE/APHE/include/aphe.h"


class dphe: public aphe
{
public:
    dphe(std::string path):aphe(path){}
    cv::Mat get_result();
private:
    double lowLimit = 0.0;
    void setLowLimit();
    cv::Mat clip_hist();
    cv::Mat get_accumulate_hist();
};
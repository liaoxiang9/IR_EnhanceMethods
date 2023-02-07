#ifndef _IR_VIS_FUSE_H_
#define _IR_VIS_FUSE_H_

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <string>


class ir_vis_fuse
{
public:
    ir_vis_fuse(const std::string &ir_path, const std::string &vis_paht): ir_img(cv::imread(ir_path)), vis_img(cv::imread(vis_paht)) {
        if (ir_img.empty() || vis_img.empty()) {
            std::cout << "Error: image is empty" << std::endl;
            exit(1);
        }
    }
    ~ir_vis_fuse(){
        std::cout << "ir_vis_fuse release"<< std::endl;
    };
    cv::Mat  ir_vis_fuse_process(const double alpha, const int threshold1, const int threshold2);
private:
    cv::Mat ir_img;
    cv::Mat vis_img;
};
#endif
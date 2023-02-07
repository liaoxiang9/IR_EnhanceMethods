#include "ir_vis_fush.h"
#include <iostream>

cv::Mat
ir_vis_fuse::ir_vis_fuse_process(
    const double alpha,
    const int threshold1, const int threshold2) 
{
    // ir 和 vis 用 clahe 进行增强
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    // 分通道进行
    cv::Mat ir_bgr[3];
    cv::Mat vis_bgr[3];
    cv::split(ir_img, ir_bgr);
    cv::split(vis_img, vis_bgr);
    for (int i = 0; i < 3; i++) {
        clahe->apply(ir_bgr[i], ir_bgr[i]);
        clahe->apply(vis_bgr[i], vis_bgr[i]);
    }
    cv::Mat ir_img_clahe;
    cv::Mat vis_img_clahe;
    cv::merge(ir_bgr, 3, ir_img_clahe);
    cv::merge(vis_bgr, 3, vis_img_clahe);

    // 融合
    cv::Mat ir_vis_fuse;
    cv::addWeighted(ir_img_clahe, alpha, vis_img_clahe, 1 - alpha, 0, ir_vis_fuse);

    // canny 检测
    cv::Mat ir_img_canny;
    cv::Mat vis_img_canny;
    cv::Canny(ir_img_clahe, ir_img_canny, threshold1, threshold2);
    cv::Canny(vis_img_clahe, vis_img_canny, threshold1, threshold2);

    // 将ir_img_canny和vis_img_canny转为float类型
    cv::Mat ir_img_canny_float;
    cv::Mat vis_img_canny_float;
    ir_img_canny.convertTo(ir_img_canny_float, CV_32FC1);
    vis_img_canny.convertTo(vis_img_canny_float, CV_32FC1);
    
    // 边缘相加
    cv::Mat edge;
    edge = ir_img_canny + vis_img_canny;

    // 在ir_vis_fuse的第三通道上加上边缘
    cv::Mat ir_vis_fuse_bgr[3];
    cv::split(ir_vis_fuse, ir_vis_fuse_bgr);
    ir_vis_fuse_bgr[2].convertTo(ir_vis_fuse_bgr[2], CV_32FC1);
    cv::add(ir_vis_fuse_bgr[2], edge, ir_vis_fuse_bgr[2], cv::noArray(), CV_32FC1);
    ir_vis_fuse_bgr[2].convertTo(ir_vis_fuse_bgr[2], CV_8UC1);
    cv::merge(ir_vis_fuse_bgr, 3, ir_vis_fuse);
    return ir_vis_fuse;
}

#include <iostream>
#include "IR_filter.h"
int main()
{
    IR_filter filter("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    // cv::Mat dst = filter.LaplacianOfGaussianFilter(filter.src, 5, 1.4);
    // cv::imwrite("/home/lx/IR_Enhance_Methods/filter/result/1_LoG.jpg", dst);
    // cv::Mat dst = filter.LocalEdgePreservingFilter(filter.src, 0.1, 1.0, 4, false);
    // cv::imwrite("/home/lx/IR_Enhance_Methods/filter/result/1_LEPF.jpg", dst);
    // cv::Mat dst = filter.GuidedFilter(filter.src, filter.src, 3, 0.001);
    // cv::imwrite("/home/lx/IR_Enhance_Methods/filter/result/1_Guided.jpg", dst);
    // cv::imshow("dst", dst);
    // cv::waitKey(0);
    cv::Mat dst = filter.BiExponentialFilter(filter.src, 0.1, 10);
    cv::imwrite("/home/lx/IR_Enhance_Methods/filter/result/1_BEEPS.jpg", dst);
    return 0;
}
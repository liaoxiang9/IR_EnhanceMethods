#include "dphe.h"

int main()
{
    dphe dphe("/home/lx/IR_EnhanceMethods/src_images/1.jpg");
    cv::Mat result = dphe.get_result();
    // write the result
    cv::imwrite("/home/lx/IR_EnhanceMethods/src_images/1_dphe_result.jpg", result);
    // cv::imshow("result", result);
    // cv::waitKey(0);
    return 0;
}
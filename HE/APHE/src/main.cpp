#include "aphe.h"

int main()
{
    aphe aphe("/home/lx/IR_EnhanceMethods/src_images/1.jpg");
    cv::Mat result = aphe.get_result();
    // cv::imshow("result", result);
    // cv::waitKey(0);
    // write result to file
    cv::imwrite("/home/lx/IR_EnhanceMethods/src_images/1_aphe_result.jpg", result);
    return 0;
}
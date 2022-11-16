#include "dphe.h"

int main()
{
    dphe dphe("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    cv::Mat result = dphe.get_result();
    cv::imshow("result", result);
    cv::waitKey(0);
    return 0;
}
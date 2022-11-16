#include "aphe.h"

int main()
{
    aphe aphe("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    cv::Mat result = aphe.get_result();
    cv::imshow("result", result);
    cv::waitKey(0);
    return 0;
}
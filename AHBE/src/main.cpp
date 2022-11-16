#include "ahbe.h"

int main(){
    cv::Mat dst;
    ahbe ahbe("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    dst = ahbe.mainProcess();
    cv::imwrite("/home/lx/IR_Enhance_Methods/AHBE/result/1_dst.jpg", dst);
    cv::imshow("dst", dst);
    // cv::imshow("dst", dst);
    cv::waitKey(0);
    // save dst
    // cv::imwrite("./result/1_dst.jpg", dst);
    return 0;
}
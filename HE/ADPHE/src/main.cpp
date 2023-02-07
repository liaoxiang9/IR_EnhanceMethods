#include "adphe.h"

int main(){
    adphe adphe("/home/lx/IR_EnhanceMethods/src_images/1.jpg");
    cv::Mat dst = adphe.mainProcess();
    cv::imwrite("/home/lx/IR_EnhanceMethods/src_images/1_adphe_result.jpg", dst);
    return 0;
}
#include "adphe.h"

int main(){
    adphe adphe("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    cv::Mat dst = adphe.mainProcess();
    cv::imwrite("1.jpg", dst);
    return 0;
}
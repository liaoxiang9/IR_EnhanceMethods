#include "contrast_stretch.h"

int main(){
    contrast_stretch c("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    cv::Mat a = c.contrast_ratio();
    cv::imshow("test", a);
    cv::waitKey(0);
}

#include "phe.h"

int main(){
    phe p("/home/lx/IR_EnhanceMethods/src_images/1.jpg");
    double clip_limit = 0.0;
    std::cout<<"please input clip_limit"<<std::endl;
    std::cin>>clip_limit;
    cv::Mat result = p.get_result(clip_limit);
    // write the result
    cv::imwrite("/home/lx/IR_EnhanceMethods/src_images/1_phe_result.jpg", result);
    // cv::imshow("result", result);
    // cv::waitKey(0);
}
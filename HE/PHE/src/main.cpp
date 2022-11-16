#include "phe.h"

int main(){
    phe p("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    double clip_limit = 0.0;
    std::cout<<"please input clip_limit"<<std::endl;
    std::cin>>clip_limit;
    cv::Mat result = p.get_result(clip_limit);
    cv::imshow("result", result);
    cv::waitKey(0);
}
#include "agcch.h"

int main(){
    std::string imagePath = "/home/lx/IR_Enhance_Methods/src_images/1.jpg";
    agcch agc(imagePath, 5);
    cv::Mat dst = agc.ielagc();
    // judge dst is same as src
    // cv::Mat src = agc.getSrc();
    // cv::Mat diff = src - src;
    cv::imwrite("ielagc1.jpg", dst);
    // std::cout << "diff: " << cv::sum(diff)[0] << std::endl;
    cv::Mat dst2 = agc.iegagc();
    cv::imwrite("iegagc1.jpg", dst2);

    // // 截取src右下角300*300的区域
    // cv::Mat src_roi = src(cv::Rect(src.cols - 300, src.rows - 300, 300, 300));
    
    // cv::imwrite("srcROI.jpg", src_roi);
    // cv::imshow("dst", dst);
    // cv::waitKey(0);
    return 0;
}
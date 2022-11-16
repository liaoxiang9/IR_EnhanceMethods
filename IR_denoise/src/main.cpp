#include "nlm.h"

using namespace std;
using namespace cv;

int main(){
    NLM n1(11, 21, 10);
    cv::Mat img_noise = cv::imread("/home/lx/IR_Enhance_Methods/src_images/hc.jpg", cv::IMREAD_GRAYSCALE);
    // 加高斯噪声
    cv::Mat img_noise_gaussian;
    cv::Mat noise = cv::Mat::zeros(img_noise.size(), CV_8UC1);
    cv::randn(noise, 0, 10);
    cv::add(img_noise, noise, img_noise_gaussian);
    // 存图
    cv::imwrite("/home/lx/IR_Enhance_Methods/src_images/noise_gaussian.jpg", img_noise_gaussian);

    n1.set_image("/home/lx/IR_Enhance_Methods/src_images/noise_gaussian.jpg");
    cv::Mat dst;
    n1.denoise(dst);
    // cv::imwrite("../result/dst.jpg", dst);
    return 0;
}
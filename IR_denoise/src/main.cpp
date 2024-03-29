#include "nlm.h"

using namespace std;
using namespace cv;

int main(){
    NLM n1(11, 21, 10);
    cv::Mat img_noise = cv::imread("/home/lx/IR_EnhanceMethods/src_images/noise_gaussian.jpg", cv::IMREAD_GRAYSCALE);
    // 检测图像是否读取成功
    if(img_noise.empty()){
        std::cout << "Error: Image not found!" << std::endl;
        exit(1);
    }

    // // 加高斯噪声
    // cv::Mat img_noise_gaussian;
    // cv::Mat noise = cv::Mat::zeros(img_noise.size(), CV_8UC1);
    // cv::randn(noise, 0, 10);
    // cv::add(img_noise, noise, img_noise_gaussian);
    // // 存图
    // cv::imwrite("/home/lx/IR_EnhanceMethods/src_images/noise_gaussian.jpg", img_noise_gaussian);

    // n1.set_image("/home/lx/IR_EnhanceMethods/src_images/noise_gaussian.jpg");
    // cv::Mat dst;
    // n1.denoise(dst);
    // cv::imwrite("../result/dst.jpg", dst);

    // using opencv nonlocalmeans
    cv::Mat dst_opencv;
    cv::fastNlMeansDenoising(img_noise, dst_opencv, 10, 21, 11);
    cv::imwrite("../result/dst_opencv.jpg", dst_opencv);

    // gaussian filter
    cv::Mat dst_gaussian;
    cv::GaussianBlur(img_noise, dst_gaussian, cv::Size(5, 5), 10);
    cv::imwrite("../result/dst_gaussian.jpg", dst_gaussian);

    // median filter
    cv::Mat dst_median;
    cv::medianBlur(img_noise, dst_median, 5);
    cv::imwrite("../result/dst_median.jpg", dst_median);

    // blur
    cv::Mat dst_blur;
    cv::blur(img_noise, dst_blur, cv::Size(5, 5));
    cv::imwrite("../result/dst_blur.jpg", dst_blur);

    return 0;
}
#include "IR_SaliencyExtraction.h"
#include <cmath>


cv::Mat IR_SaliencyExtraction::frequencyTunedSaliencyMap(cv::Mat src, float alpha, float beta, float gamma, int r){
    // gaussianblur
    cv::Mat gaussianBlur;
    cv::GaussianBlur(src, gaussianBlur, cv::Size(r, r), gamma);
    // get mean of src
    cv::Scalar mean = cv::mean(src);

    // get saliency map
    cv::Mat saliencyMap = cv::Mat::zeros(src.size(), CV_32FC1);

    cv::Mat src_ = src.clone();
    src_.convertTo(src_, CV_32FC1);
    // get src-mean{[0] ^ 2
    cv::Mat srcMean = src_ - mean[0];
    cv::pow(srcMean, 2, srcMean);
    cv::sqrt(srcMean, saliencyMap);

    // normalize the saliency map
    cv::normalize(saliencyMap, saliencyMap, 0, 1, cv::NORM_MINMAX);
    // get the frequency-tuned saliency map
    saliencyMap = saliencyMap * 255;
    // saliencyMap.convertTo(saliencyMap, CV_8UC1);

    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    dst = 0.5 * saliencyMap + 0.5 * src_;

    dst.setTo(0, dst < 0);
    dst.setTo(255, dst > 255);
    dst.convertTo(dst, CV_8UC1);
    return dst;
}


cv::Mat IR_SaliencyExtraction::multiScaleFTSaliencyMap(cv::Mat src, int r, int t0, int delta_t, float alpha1, float alpha2, float sigma1, float sigma2, int n){
    cv::Mat srcPadding;
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat pre = cv::Mat::zeros(src.size(), CV_32FC1);
    // get the gaussian kernel
    cv::Mat gaussianKernel1 = cv::Mat::zeros(cv::Size(r, r), CV_32FC1);
    cv::Mat gaussianKernel2 = cv::Mat::zeros(cv::Size(r, r), CV_32FC1);
    int center = r / 2;
    for(int i = 0; i < r; i++){
        for(int j = 0; j < r; j++){
            gaussianKernel1.at<float>(i, j) = exp(-((pow(i - center, 2) + pow(j - center, 2)) / (2 * pow(sigma1, 2))));
            gaussianKernel2.at<float>(i, j) = exp(-((pow(i - center, 2) + pow(j - center, 2)) / (2 * pow(sigma2, 2))));
        }
    }
    // normalize the gaussian kernel
    gaussianKernel1 = gaussianKernel1 / cv::sum(gaussianKernel1)[0];
    gaussianKernel2 = gaussianKernel2 / cv::sum(gaussianKernel2)[0];

    for(int i=1;i<n;i++){
        int t=t0+delta_t*i;
        cv::copyMakeBorder(src, srcPadding, t, t, t, t, cv::BORDER_REFLECT);
        for(int m=0;m<src.rows;m++){
            for(int n=0;n<src.cols;n++){
                cv::Mat window = srcPadding(cv::Rect(n, m, t, t));
                dst.at<float>(m, n) = dilatedGaussianConv(window, gaussianKernel1,gaussianKernel2, r, sigma1, sigma2);
                // if(dst.at<float>(m, n) == INFINITY){
                //     dst.at<float>(m, n) = 0;
                //     std::cout << "i: " << i << " m: " << m << " n: " << n << std::endl;
                // }

            }
        }
    cv::max(dst, pre, dst);
    pre = dst.clone();
    }
    // dst.setTo(0, dst < 0);
    // dst.setTo(255, dst > 255);

    // cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    // dst = dst * 255;
    // cout the min and max of dst
    double min, max;
    cv::minMaxLoc(dst, &min, &max);
    std::cout << "min: " << min << std::endl;
    std::cout << "max: " << max << std::endl;
    cv::Mat src_ = src.clone();
    src_.convertTo(src_, CV_32FC1);

    // normalize the dst
    cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    dst = dst * 255;
    // dst = 0.0 * dst;

    
    cv::Mat result;
    // dst = cv::Mat::zeros(src.size(), CV_32FC1);
    result = alpha1 * dst + alpha2 * src_;
    // // dst.setTo(0, dst < 0);
    // // dst.setTo(255, dst > 255);
    // dst.convertTo(dst, CV_8UC1);
    
    return result;
}


float IR_SaliencyExtraction::dilatedGaussianConv(cv::Mat window, cv::Mat gaussianKernel1, cv::Mat gaussianKernel2,int r, float sigma1, float sigma2){
    cv::Mat gaussianBlurLow;
    cv::Mat gaussianBlurHigh;
    if(window.cols != window.rows){
        throw "window must be square";
    }
    // get the downSampled window
    cv::Mat downSampledWindow = cv::Mat::zeros(cv::Size(r, r), CV_32FC1);
    window.convertTo(window, CV_32FC1);
    int step = (window.cols / r) + 1;
    // std::cout << "step: " << step << std::endl;
    for(int i = 0; i < r; i++){
        for(int j = 0; j < r; j++){
            downSampledWindow.at<float>(i, j) = window.at<float>(i * step, j * step);
        }
    }
    // // get the gaussian blur of downSampledWindow
    // cv::GaussianBlur(downSampledWindow, gaussianBlurLow, cv::Size(r, r), sigma1);
    // cv::GaussianBlur(downSampledWindow, gaussianBlurHigh, cv::Size(r, r), sigma2);

    // get mean of window
    // cv::Scalar mean = cv::mean(downSampledWindow);
    // std::cout << "mean: " << mean[0] << std::endl;
    // get the dilated gaussian convolution
    float low =  twoDimensionGaussian(downSampledWindow, gaussianKernel1, r, sigma1);
    float high = twoDimensionGaussian(downSampledWindow, gaussianKernel2, r, sigma2);
    // float high = gaussianBlurHigh.at<float>(r / 2, r / 2);
    return sqrt(pow(low - high, 2));
}

// 2D Gaussian blur without opencv

float  IR_SaliencyExtraction::twoDimensionGaussian(cv::Mat window, cv::Mat gaussianKernel, int r, float sigma){

    // get the value of convolution of center pixel
    float centerValue = 0;
    for(int i = 0; i < r; i++){
        for(int j = 0; j < r; j++){
            centerValue += gaussianKernel.at<float>(i, j) * window.at<float>(i, j);
        }
    }
    return centerValue;
}
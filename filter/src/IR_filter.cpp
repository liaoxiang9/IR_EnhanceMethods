#include "IR_filter.h"
#include <cmath>
#include <opencv2/core/types.hpp>
cv::Mat IR_filter::LMS_2D(cv::Mat src, int windowSize, double stepSize, int iteration){
    // get  the reference image
    cv::Mat ref = src.clone();
    cv::Mat dst = src.clone();
    cv::Mat error = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat filter = cv::Mat::zeros(windowSize, windowSize, CV_8UC1);
    cv::Mat filterTemp = cv::Mat::zeros(windowSize, windowSize, CV_8UC1);
    cv::Mat filterTemp2 = cv::Mat::zeros(windowSize, windowSize, CV_8UC1);
    cv::Mat filterTemp3 = cv::Mat::zeros(windowSize, windowSize, CV_8UC1);
    cv::Mat filterTemp4 = cv::Mat::zeros(windowSize, windowSize, CV_8UC1);
    cv::Mat filterTemp5 = cv::Mat::zeros(windowSize, windowSize, CV_8UC1);
    cv::Mat filterTemp6 = cv::Mat::zeros(windowSize, windowSize, CV_8UC1);
    return dst;
}

float IR_filter::LoGFunction(int x, int y, double sigma){
    return (-1.0/(M_PI*sigma*sigma*sigma*sigma))*(1.0-((x*x+y*y)/(2.0*sigma*sigma)))*exp(-(x*x+y*y)/(2.0*sigma*sigma));
}


cv::Mat IR_filter::LaplacianOfGaussianFilter(cv::Mat src, int windowSize, double sigma){
    cv::Mat gaussian;
    cv::GaussianBlur(src, gaussian, cv::Size(windowSize, windowSize), sigma);
    cv::Mat dst;
    cv::Laplacian(gaussian, dst, CV_8UC1, 3, 1, 0, cv::BORDER_DEFAULT);
    return dst;
}


cv::Mat IR_filter::LocalEdgePreservingFilter(cv::Mat src,  float alpha, float beta, int windowSize, bool isFloat){
    // get the height and width of the image
    int height = src.rows;
    int width = src.cols;
    cv::Mat I = src.clone();
    // convert I to float
    I.convertTo(I, CV_32FC1);
    // get N
    cv::Mat N;
    cv::boxFilter(cv::Mat::ones(src.size(), CV_32FC1), N, -1, cv::Size(windowSize, windowSize));
    // get mean_I
    cv::Mat mean_I = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(I, mean_I, -1, cv::Size(windowSize, windowSize));
    // get mean_II
    cv::Mat mean_II = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(I.mul(I), mean_II, -1, cv::Size(windowSize, windowSize));
    // get var_I
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);

    // 进行列方向差分，求dx
    cv::Mat dx = cv::Mat::zeros(src.size(), CV_32FC1);
    for(int i=0; i<height; i++){
        for(int j=0; j<width-1; j++){
            dx.at<float>(i, j) = I.at<float>(i, j+1) - I.at<float>(i, j);
        }
    }
    // 进行行方向差分，求dy
    cv::Mat dy = cv::Mat::zeros(src.size(), CV_32FC1);
    for(int i=0; i<height-1; i++){
        for(int j=0; j<width; j++){
            dy.at<float>(i, j) = I.at<float>(i+1, j) - I.at<float>(i, j);
        }
    }
    // get I_grad
    cv::Mat I_grad = cv::Mat::zeros(src.size(), CV_32FC1);
    I_grad = cv::abs(dx+dx);
    cv::pow(I_grad, 2-beta, I_grad);
    cv::Mat tmp = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(I_grad, tmp, -1, cv::Size(windowSize, windowSize));
    I_grad = alpha * tmp / N;
    // get a
    cv::Mat a = cv::Mat::zeros(src.size(), CV_32FC1);
    a = var_I / (var_I + I_grad + 1e-8);
    // get b
    cv::Mat b = cv::Mat::zeros(src.size(), CV_32FC1);
    b = mean_I - a.mul(mean_I);
    // get mean_a
    cv::Mat mean_a = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(a, mean_a, -1, cv::Size(windowSize, windowSize));
    mean_a =  mean_a / N;
    // get mean_b
    cv::Mat mean_b = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(b, mean_b, -1, cv::Size(windowSize, windowSize));
    mean_b = mean_b / N;
    // get dst
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    dst = mean_a.mul(I) + mean_b;
    // convert dst to CV_8UC1
    if(!isFloat){
        dst.convertTo(dst, CV_8UC1);
        return dst;
    }else{
        return dst;
    }
}

cv::Mat IR_filter::GuidedFilter(cv::Mat src, cv::Mat guide, int windowSize, double eps){
    // 引导滤波
    cv::Mat mean_I = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(guide, mean_I, -1, cv::Size(windowSize, windowSize));
    cv::Mat mean_p = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(src, mean_p, -1, cv::Size(windowSize, windowSize));
    cv::Mat mean_Ip = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(guide.mul(src), mean_Ip, -1, cv::Size(windowSize, windowSize));
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    cv::Mat mean_II = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(guide.mul(guide), mean_II, -1, cv::Size(windowSize, windowSize));
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);
    cv::Mat a = cv::Mat::zeros(src.size(), CV_32FC1);
    a = cov_Ip / (var_I + eps);
    cv::Mat b = cv::Mat::zeros(src.size(), CV_32FC1);
    b = mean_p - a.mul(mean_I);
    cv::Mat mean_a = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(a, mean_a, -1, cv::Size(windowSize, windowSize));
    cv::Mat mean_b = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::boxFilter(b, mean_b, -1, cv::Size(windowSize, windowSize));
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    dst = mean_a.mul(guide) + mean_b;
    return dst;
}

cv::Mat IR_filter::BilateralFilter(cv::Mat src, int windowSize, double sigmaColor, double sigmaSpace){
    // 双边滤波
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::bilateralFilter(src, dst, windowSize, sigmaColor, sigmaSpace);
    return dst;
}

cv::Mat IR_filter::BiExponentialFilter(cv::Mat src, float sigma, float lambda){
    cv::Mat vertical = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat horizontal = cv::Mat::zeros(src.size(), CV_32FC1);
    vertical = BEEPS(src, sigma, lambda, true);
    horizontal = BEEPS(src, sigma, lambda, false);
    // convert vertical and horizontal to CV_32FC1
    // vertical.convertTo(vertical, CV_32FC1);
    // horizontal.convertTo(horizontal, CV_32FC1);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    dst = (vertical + horizontal)/2;
    // // // set the value of dst = 255 if dst >255
    // dst.setTo(cv::Scalar(255), dst>255);
    // dst.setTo(cv::Scalar(0), dst<0);

    // // // 输出dst 的最大值和最小值
    // double minVal, maxVal;
    // cv::minMaxLoc(horizontal, &minVal, &maxVal);
    // std::cout << "minVal = " << minVal << std::endl;
    // std::cout << "maxVal = " << maxVal << std::endl;
    // convert dst to CV_8UC1
    dst.convertTo(dst, CV_8UC1);
    return dst;
}


float IR_filter::gaussianFunction(float x, float y, float sigma){
    return exp(-((x-y)*(x-y))/(2*sigma*sigma));
}


cv::Mat IR_filter::BEEPS(cv::Mat src, float sigma, float lambda, bool isVertical){
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    // BEEPS implementation
    int height = src.rows;
    int width = src.cols;
    int size = height * width;
    cv::Mat src_flat = cv::Mat::zeros(size, 1, CV_32FC1);
    cv::Mat psi = cv::Mat::zeros(size, 1, CV_32FC1);
    cv::Mat phi = cv::Mat::zeros(size, 1, CV_32FC1);
    cv::Mat y = cv::Mat::zeros(size, 1, CV_32FC1);
    if(!isVertical){
         // 将图像按行展开
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                src_flat.at<float>(i*width+j, 0) = src.at<uchar>(i, j);
            }
        }
        psi.at<float>(size-1, 0) = src_flat.at<float>(size-1, 0);
        phi.at<float>(0, 0) = src_flat.at<float>(0, 0);
        for(int i=size-2; i>=0; i--){
            psi.at<float>(i, 0) = gaussianFunction(src_flat.at<float>(i, 0), psi.at<float>(i+1, 0), sigma);
        }
        for(int i=1; i<size; i++){
            phi.at<float>(i, 0) = gaussianFunction(src_flat.at<float>(i, 0), phi.at<float>(i-1, 0), sigma);
        }
        for(int i=0; i<size; i++){
            y.at<float>(i, 0) = (phi.at<float>(i, 0) + psi.at<float>(i, 0) - (1-lambda)*src_flat.at<float>(i, 0)) / (1+lambda);
        }
    // reshape y to dst
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                dst.at<float>(i, j) = y.at<float>(i*width+j, 0);
            }
        }
        return dst;
    }else{
        // 将图像按列展开
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                src_flat.at<float>(j*height+i, 0) = src.at<uchar>(i, j);
            }
        }
        psi.at<float>(size-1, 0) = src_flat.at<float>(size-1, 0);
        phi.at<float>(0, 0) = src_flat.at<float>(0, 0);
        for(int i=size-2; i>=0; i--){
            psi.at<float>(i, 0) = gaussianFunction(src_flat.at<float>(i, 0), psi.at<float>(i+1, 0), sigma);
        }
        for(int i=1; i<size; i++){
            phi.at<float>(i, 0) = gaussianFunction(src_flat.at<float>(i, 0), phi.at<float>(i-1, 0), sigma);
        }
        for(int i=0; i<size; i++){
            y.at<float>(i, 0) = (phi.at<float>(i, 0) + psi.at<float>(i, 0) - (1-lambda)*src_flat.at<float>(i, 0)) / (1+lambda);
        }

        // reshape y to dst
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                dst.at<float>(i, j) = y.at<float>(j*height+i, 0);
            }
        }
        // convert dst to CV_8UC1
        // dst.convertTo(dst, CV_8UC1);
        return dst;
    }
}


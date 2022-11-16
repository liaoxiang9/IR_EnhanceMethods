#include "ahbe.h"
#include <cmath>
#include <iostream>

// Language: cpp
// Path: src/ahbe.cpp
// Compare this snippet from src/ahbe.cpp:
// #include "ahbe.h"

int ahbe::adaptiveThresholdSelection(cv::Mat &hist){
    // random select an initial threshold between 0 and 255(not included)
    int threshold0 = rand() % 256;
    while(threshold0 == 0 || threshold0 == 255){
        threshold0 = rand() % 256;
    }
    int threshold1 = 256;
    bool flag = true;
    while(flag){
        // calculate the mean of the background and foreground
        float sum0 = 0.0;
        float sum1 = 0.0;
        float count0 = 0.0;
        float count1 = 0.0;
        for(int i = 0; i < 256; i++){
            if(i < threshold0){
                sum0 += i * hist.at<float>(i);
                count0 += hist.at<float>(i);
            }
            else{
                sum1 += i * hist.at<float>(i);
                count1 += hist.at<float>(i);
            }
        }
        float mean0 = sum0 / count0;
        float mean1 = sum1 / count1;
        threshold1 = (int)round((mean0 + mean1) / 2);
        // std::cout << "threshold0: " << threshold0 << std::endl;
        std::cout << "threshold: " << threshold1 << std::endl;
        // judge whether the threshold is converged
        if(threshold1 == threshold0){
            flag = false;
        }
        else{
            threshold0 = threshold1;
        }
    }
    return threshold1;
}

cv::Mat ahbe::hch(cv::Mat& hist, int threshold){
    // normalize the histogram
    hist /= (src.rows * src.cols);

    cv::Mat target_pdf = hist.clone();
    float sum1 = 0.0;
    float sum2 = 0.0;
    float p_th = hist.at<float>(threshold);
    for(int i=0;i<256;i++){
        if(i<threshold){
            for(int j=0;j<i;j++){
                sum1 += std::min(hist.at<float>(j), p_th);
            }
            target_pdf.at<float>(i) = sum1;
            sum1 = 0.0;
        }
        else{
            for(int j=threshold;j<i;j++){
                sum2 += hist.at<float>(j) * (log(j) - log(2));
            }
            target_pdf.at<float>(i) = sum2;
            sum2 = 0.0;
        }
    }
    // calculate the cdf of the target pdf
    cv::Mat target_cdf = target_pdf.clone();
    for(int i=1;i<256;i++){
        target_cdf.at<float>(i) += target_cdf.at<float>(i-1);
    }  
}


void ahbe::getHist(){
    // calculate the histogram of the image
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange[] = {range};
    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, true, false);
}

cv::Mat ahbe::getHBFKernel(int G){
    // get the high-boost filter kernel
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_32FC1);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(i==1 && j==1){
                kernel.at<float>(i, j) = 4 + G;
            }
            else if(i==1 || j==1){
                kernel.at<float>(i, j) = -1;
            }
            else{
                kernel.at<float>(i, j) = 0;
            }
        }
    }
    return kernel;
}


cv::Mat ahbe::mainProcess(){
    // get the histogram of the image
    getHist();
    // adaptive threshold selection
    int threshold = adaptiveThresholdSelection(hist);
    // histogram equalization
    cv::Mat hist_equalized = hch(hist, threshold);
    // high-boost filtering
    cv::Mat kernel = getHBFKernel(1);
    // output value of kernel
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            std::cout << (int)kernel.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }

    cv::Mat result;
    cv::filter2D(src, result, -1, kernel);
    // return the result
    return result;
}
#ifndef _APHE_H_
#define _APHE_H_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>


class aphe
{
public:
    cv::Mat src;
    aphe(const std::string &filename): src(cv::imread(filename, cv::IMREAD_GRAYSCALE)) {
        if (src.empty()) {
            std::cout << "Error: image is empty" << std::endl;
            exit(1);
        }
    }
    ~aphe(){
        std::cout << "aphe object is destroyed" << std::endl;
    }
    void set(const std::string &filename){
        src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if(src.empty()){
            std::cout << "Error: image is empty" << std::endl;
            exit(1);
        }
    }
    cv::Mat get_result() {
        cv::Mat accumulate_hist = get_accumulate_hist();
        cv::Mat result = src.clone();
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                result.at<uchar>(i, j) = (uchar)accumulate_hist.at<float>(src.at<uchar>(i, j));
            }
        }
        return result;
    }

    double getThreshold() {
        setThreshhold();
        return threshhold;
    }

protected:
    
    double threshhold = 0.0;
    const int histSize = 256;
    // find median value of 7 numbers
    float find_median(std::vector<double> &vec) {
        std::sort(vec.begin(), vec.end());
        return vec[3];
    }

    void floor_(cv::Mat& src_hist){
        for(int i=0;i<histSize;i++){
            src_hist.at<float>(i) = floor(src_hist.at<float>(i));
        }
    }


    // median filter of 3 neighbors
    cv::Mat medianFilter(const cv::Mat &hist){
        cv::Mat medianFilter = hist.clone();
        std::vector<double> vec(7);
        for(int i = 3; i < hist.rows - 3; i++){
            for(int j=0;j<7;j++){
                vec[j] = hist.at<float>(i-3+j);
            }
            medianFilter.at<float>(i) = find_median(vec);
        }
        return medianFilter;
    }


    void setThreshhold(){
        cv::Mat hist = getHist();
        // std::cout << "rows:"<<hist.rows << std::endl;
        // for(int i=0;i<histSize;i++){
        //     std::cout<<hist.at<float>(i)<<std::endl;
        // }

        cv::Mat medianResult = medianFilter(hist);
        int sum = 0;
        for(int i = 0;i<medianResult.rows;i++){
            if (medianResult.at<float>(i) > 0) {
                // std::cout << "medianResult.at<float>(i) = " << medianResult.at<float>(i) << std::endl;
                sum += 1;
            };
        }
        cv::Mat F = cv::Mat::zeros(sum, 1, CV_64FC1);
        std::cout << "sum=" << sum << std::endl;
        
        int index = 0;
        for(int i = 0;i<medianResult.rows;i++){
            if (medianResult.at<float>(i) > 0){
                // std::cout << "medianResult.at<double>(i) = " << medianResult.at<double>(i) << std::endl;
                F.at<float>(index) = medianResult.at<float>(i);
                index++;
            }
        }
        // std::cout << "rows=" << F.rows << std::endl;
        // std::cout << "index=" <<index << std::endl;


        cv::Mat F1 = F.clone();
        for(int i =1;i<F.rows;i++){ 
            F1.at<float>(i) = F.at<float>(i) - F.at<float>(i - 1);
            // std::cout << "F1.at<float>(index) = " << F1.at<float>(i) << std::endl;

        }

        std::vector<float> F1Vec;
        for(int i =1;i<F1.rows - 1;i++){
            if(std::abs(F1.at<float>(i)) < std::min(std::abs(F1.at<float>(i-1)), std::abs(F1.at<float>(i+1))) 
                     || (F1.at<float>(i - 1) < 0 && F1.at<float>(i + 1) > 0)){
                F1Vec.push_back(F.at<float>(i));
                // std::cout << "F1.at<float>(index) = " << F1.at<float>(i) << std::endl;

            }
            // F1Vec.push_back(F1.at<float>(i));
            // std::cout << F1.at<double>(i) << std::endl;
        }

        std::cout << "F1Vec.size() = " << F1Vec.size() << std::endl;

        // sort first
        std::sort(F1Vec.begin(), F1Vec.end());
        // find median value
        if(F1Vec.size() % 2 == 0){
            threshhold = (F1Vec[F1Vec.size() / 2] + F1Vec[F1Vec.size() / 2 - 1]) / 2;
        }
        else{
            threshhold = F1Vec[F1Vec.size() / 2];
        }
        std::cout << "threshhold: " << threshhold << std::endl;
    }

    cv::Mat getHist(){
        cv::Mat hist;
        float range[] = { 0, 256 };
        const float* histRange[] = { range }; // 指针数组：数组元素是指针的数组，也就是个二维指针
        bool uniform = true, accumulate = false;
        cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate);
        cv::Scalar s = cv::sum(hist);
        hist /= s[0];

        return hist;
    }

    cv::Mat clip_hist() {
        cv::Mat hist_ = getHist();
        
        cv::Mat clipped_hist = hist_.clone();
        setThreshhold();

        clipped_hist.setTo(threshhold, clipped_hist > threshhold);
        
        return clipped_hist;
    }


    // get accumulate of clipped hist
    cv::Mat get_accumulate_hist() {
        cv::Mat clipped_hist = clip_hist();

        cv::Mat accumulate_hist = clipped_hist.clone();
        // 计算累计直方图
        for (int i = 1; i < histSize; i++) {
            accumulate_hist.at<float>(i) += accumulate_hist.at<float>(i - 1);
        }
        // 归一化并乘以255再向下取整
        std::cout <<"acc sum is:"<< accumulate_hist.at<float>(histSize - 1) << std::endl;
        accumulate_hist= accumulate_hist * (histSize - 1) / accumulate_hist.at<float>(histSize - 1);
        floor_(accumulate_hist);
        return accumulate_hist;
    }


};

#endif
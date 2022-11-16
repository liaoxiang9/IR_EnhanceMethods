#ifndef _PHE_H_
#define _PHE_H_
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <cmath>
#include <iostream>


class phe{
public:
    phe(const std::string& img_path) : src_img(cv::imread(img_path, cv::IMREAD_GRAYSCALE)) {
        if (src_img.empty()) {
            throw std::runtime_error("Can't open image");
        }
    }
    ~phe(){
        std::cout<<"GG"<<std::endl;
    };
    void set(const std::string& img_path);

        // get result 
    cv::Mat get_result(double clip_limit) {
        cv::Mat accumulate_hist = get_accumulate_hist(clip_limit);
        cv::Mat result = src_img.clone();
        // 只能遍历吗？
        for (int i = 0; i < src_img.rows; i++) {
            for (int j = 0; j < src_img.cols; j++) {
                result.at<uchar>(i, j) = (uchar)accumulate_hist.at<float>(src_img.at<uchar>(i, j));
            }
        }
        return result;
    }

private:
    cv::Mat src_img;
    const int histSize = 256;
    // get hist of image
    // 返回的还是float类型
    void floor_(cv::Mat& src_hist){
        for(int i=0;i<histSize;i++){
            src_hist.at<float>(i) = floor(src_hist.at<float>(i));
        }
    }


    cv::Mat get_hist() {
        cv::Mat hist;
        float range[] = { 0, 256 };
        const float* histRange[] = { range }; // 指针数组：数组元素是指针的数组，也就是个二维指针
        bool uniform = true, accumulate = false;
        cv::calcHist(&src_img, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate);
        cv::Scalar s = cv::sum(hist);
        std::cout << "sum is " << s[0] << std::endl;
        hist /= s[0];
        for(int i=0;i<histSize;i++){
            std::cout<<hist.at<float>(i)<<std::endl;
        }
        return hist;

    }

    cv::Mat clip_hist(double clip_limit) {
        cv::Mat hist = get_hist();
        cv::Mat clipped_hist = hist.clone();
        clipped_hist.setTo(clip_limit, clipped_hist > clip_limit);
        return clipped_hist;
    }


    // get accumulate of clipped hist
    cv::Mat get_accumulate_hist(double clip_limit) {
        cv::Mat clipped_hist = clip_hist(clip_limit);
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
#include "cowc.h"

cv::Mat cowc::DWT(cv::Mat& src, int level){
   // get the width and height of the image
    int width = src.cols;
    int height = src.rows;
    cv::Mat img_l = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat img_h = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat img_l_l = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat img_l_h = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat img_h_l = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat img_h_h = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat imgNew = cv::Mat::zeros(height, width, CV_32FC1);

    int depthCount = 0;
    while(depthCount<level){
        // haar decomposition
        if(depthCount!=0){
            width = width/2;
            height = height/2;
        }
        // 行滤波
        // 行间滤波，再进行下采样
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                if(j%2==0){
                    img_l.at<float>(i, j/2) = (src.at<float>(i, j)+src.at<float>(i, j+1))/2;
                    img_h.at<float>(i, j/2) = (src.at<float>(i, j)-src.at<float>(i, j+1))/2;
                }
            }
        }
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                if(i%2==0 && j%2==0){
                    img_l_l.at<float>(i,j) = (src.at<float>(i,j)+src.at<float>(i+1,j)+src.at<float>(i,j+1)+src.at<float>(i+1,j+1))/4;
                    img_l_h.at<float>(i,j) = (src.at<float>(i,j)-src.at<float>(i+1,j)+src.at<float>(i,j+1)-src.at<float>(i+1,j+1))/4;
                    img_h_l.at<float>(i,j) = (src.at<float>(i,j)+src.at<float>(i+1,j)-src.at<float>(i,j+1)-src.at<float>(i+1,j+1))/4;
                    img_h_h.at<float>(i,j) = (src.at<float>(i,j)-src.at<float>(i+1,j)-src.at<float>(i,j+1)+src.at<float>(i+1,j+1))/4;
                }
            }
        }
        // copy the haar decomposition result to the original image
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                if(i%2==0 && j%2==0){
                    src.at<float>(i,j) = img_l_l.at<float>(i,j);
                    src.at<float>(i+1,j) = img_l_h.at<float>(i,j);
                    src.at<float>(i,j+1) = img_h_l.at<float>(i,j);
                    src.at<float>(i+1,j+1) = img_h_h.at<float>(i,j);
                }
            }
        }
        depthCount++;
        height = height/2;
        width = width/2;
    }

    }
}

#ifndef _IR_SALIENCY_EXTRACTION_H_
#define _IR_SALIENCY_EXTRACTION_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
class IR_SaliencyExtraction
{
public:
    IR_SaliencyExtraction(std::string path) : src(cv::imread(path, 0))
    {
        if (src.empty())
        {
            std::cout << "Image is empty!" << std::endl;
        }
    }
    ~IR_SaliencyExtraction()
    {
        std::cout << "IR_SaliencyExtraction is destroyed!" << std::endl;
    };
    cv::Mat frequencyTunedSaliencyMap(cv::Mat src, float alpha, float beta, float gamma, int r);
    cv::Mat multiScaleFTSaliencyMap(cv::Mat src, int r, int t0, 
                                    int delta_t, float alpha1, float alpha2, 
                                    float sigma1, float sigma2, int n);
    cv::Mat getSrc(){
        return src;
    }

private:
    cv::Mat src;
    float dilatedGaussianConv(cv::Mat window, cv::Mat  gaussianKernel1, cv::Mat  gaussianKernel2, int r, float sigma1, float sigma2);
    float twoDimensionGaussian(cv::Mat window, cv::Mat gaussianKernel, int r, float sigma);
};

#endif
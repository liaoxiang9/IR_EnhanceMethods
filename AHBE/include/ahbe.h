#ifndef _AHBE_H_
#define _AHBE_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

class ahbe
{
  public:
    ahbe(const std::string &filename): src(cv::imread(filename, 0)) {
        if (src.empty()) {
            std::cout << "Error: image is empty" << std::endl;
            exit(1);
        }
    };
    ~ahbe() {
        std::cout << "ahbe object is destroyed" << std::endl;
    };
    cv::Mat mainProcess();
  protected:
    cv::Mat src;
    cv::Mat hist;
    
    int adaptiveThresholdSelection(cv::Mat &hist);

    cv::Mat hch(cv::Mat&, int);

    void getHist();

    // get high-boost filter kernel
    cv::Mat getHBFKernel(int);
};



#endif
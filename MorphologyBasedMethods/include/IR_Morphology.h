#ifndef _IR_Morphology_H_
#define _IR_Morphology_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>

#define blackTopHatType 0
#define whiteTopHatType 1
class IR_Morphology
{

    public:

        IR_Morphology(std::string srcPath):src(cv::imread(srcPath, 0)){
            if(src.empty()){
                std::cout << "Image is empty!" << std::endl;
            }
        };
        ~IR_Morphology(){
            std::cout<<"IR_Morphology is destroyed!"<<std::endl;
        };

        cv::Mat getSrc(){
            return src;
        };

        cv::Mat whiteTopHat(cv::Mat src, int kernelSize);
        cv::Mat blackTopHat(cv::Mat src, int kernelSize);
        
        /*
        * @param L: the size of the window
        * @param delta: the coefficient of the threshold
        * @param alpha: the coefficient to control the portion of the src
        * @param beta: the threshold of control the MWTH
        * @param gamma: the threshold of control the MBTH
        * @return dst: the modified top hat image
        */
        cv::Mat modifiedTopHat(int L, float delta, float alpha, float beta, float gamma);

        /*
        * @param L: the size of the window
        * @param n: the number of the loop
        * @param W: the size of  the outer window
        * @param M: the size of the border of the outer window
        * @param alpha: the coefficient to control the portion of the src
        * @param beta: the coefficient of control the NWTH
        * @param gamma: the coefficient of control the NBTH
        * @param n_S: the step of the window
        */
        cv::Mat multiScaleTopHat(int L, int n, int W, int M, float alpha, float beta, float gamma, int n_S);
        

        /*
        * @param kernelSize: the size of the kernel
        * @param threshold: the threshold of the kernel
        */
        cv::Mat morphologyCenterOperator(int kernelSize, int threshold);

        // print a Mat like a matrix
        void printMat(const cv::Mat& src);
    private:
        cv::Mat src;
        cv::Mat morphologyCenterOperator_(cv::Mat src, int kernelSize);
};

#endif
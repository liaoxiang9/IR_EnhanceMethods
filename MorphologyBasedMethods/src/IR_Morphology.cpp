#include "IR_Morphology.h"

void IR_Morphology::printMat(const cv::Mat& src){
    // get the type of the matrix
    int type = src.type();
    // 遍历
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            // get the value of the matrix
            if(type == CV_8UC1){
                std::cout << (int)src.at<uchar>(i, j) << " ";
            }else if(type == CV_8UC3){
                std::cout << (int)src.at<cv::Vec3b>(i, j)[0] << " ";
            }else if(type == CV_32FC1){
                std::cout << src.at<float>(i, j) << " ";
            }
        }
        std::cout << std::endl;
    }
}

cv::Mat IR_Morphology::whiteTopHat(cv::Mat src, int kernelSize){
    cv::Mat dst;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(src, dst, cv::MORPH_TOPHAT, kernel);
    return dst;
}

cv::Mat IR_Morphology::blackTopHat(cv::Mat src, int kernelSize){
    cv::Mat dst;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(src, dst, cv::MORPH_BLACKHAT, kernel);
    return dst;
}

cv::Mat IR_Morphology::modifiedTopHat(int L, float delta, float alpha, float beta, float gamma){
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    // Select a L*L window w, locate the centre of w at each pixel of src
    // padding the src with L/2 pixels on each side 
    cv::Mat srcPad;
    cv::copyMakeBorder(src, srcPad, L/2, L/2, L/2, L/2, cv::BORDER_CONSTANT, cv::Scalar(0));
    // get the size of the src
    int rows = src.rows;
    int cols = src.cols;
    // get the size of the srcPad
    int rowsPad = srcPad.rows;
    int colsPad = srcPad.cols;
    // initial the GCM
    cv::Mat GCM = cv::Mat::zeros(rows, cols, CV_32FC1);
    // over the srcPad
    for(int i = L/2; i < rowsPad - L/2; i++){
        for(int j = L/2; j < colsPad - L/2; j++){
            // get thr L*L window around the pixel
            cv::Mat window = srcPad(cv::Rect(j - L/2, i - L/2, L, L));
            // get the max value and min value of the window
            double maxVal, minVal;
            cv::minMaxLoc(window, &minVal, &maxVal);
            // get the GCM
            GCM.at<float>(i - L/2, j - L/2) = maxVal - minVal;
            // 窗口代表了一个区域，区域内的像素值的最大值和最小值的差值就是GCM
            // 代表了区域内的像素值的变化程度
            // get the modeified top-hat image

        }
    }
    // calculate the mean and variance of the GCM
    cv::Scalar mean, stddev;
    cv::meanStdDev(GCM, mean, stddev);

    // calculate the threshold
    float threshold = mean[0] + delta * stddev[0];
    // cout threshold
    std::cout << "threshold: " << threshold << std::endl;
    cv::Mat MWTH = cv::Mat::zeros(rows, cols, CV_32FC1);    
    cv::Mat MBTH = cv::Mat::zeros(rows, cols, CV_32FC1);
    MWTH = whiteTopHat(src, L);
    cv::max(MWTH, threshold, MWTH);
    MWTH = MWTH - threshold;
    MBTH = blackTopHat(src, L);
    cv::max(MBTH, threshold, MBTH);
    MBTH = MBTH - threshold;
    cv::Mat src_ = src.clone();
    src_.convertTo(src_, CV_32FC1);
    MWTH.convertTo(MWTH, CV_32FC1);
    MBTH.convertTo(MBTH, CV_32FC1);
    dst = alpha*src_ + beta*MWTH - gamma*MBTH;
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

cv::Mat IR_Morphology::multiScaleTopHat(int L, int n, int W, int M, float alpha, float beta, float gamma, int n_S){
    // initial the dst
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    // 构造结构元
    cv::Mat kernel_Delta;
    cv::Mat kernel_b;
    // 创建全0矩阵
    cv::Mat RW = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat RB = cv::Mat::zeros(src.size(), CV_32FC1);  
    cv::Mat src_ = src.clone();

    // initial
    cv::Mat dilation;
    cv::Mat erosion;
    cv::Mat close_;
    cv::Mat open_;
    cv::Mat NWTH;
    cv::Mat erosion_b;
    cv::Mat dilation_b;
    cv::Mat NBTH;
    src_.convertTo(src_, CV_32FC1);
    for(int i=1;i<=n;i++){
        int nL_s = L + i * n_S;
        int nW_s = W +  i * n_S;
        // 构造结构元
        kernel_Delta = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(nW_s, nW_s));
        kernel_b = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(nL_s, nL_s));
        // make the kernel_Delta inner M*M pixels to 0
        for(int j = 0; j < nW_s; j++){
            for(int k = 0; k < nW_s; k++){
                if(j >= (nW_s - M)/2 && j < (nW_s + M)/2 && k >= (nW_s - M)/2 && k < (nW_s + M)/2){
                    kernel_Delta.at<uchar>(j, k) = 0;
                }
            }
        }
        // get dilation of src using kernel_Delta

        cv::dilate(src, dilation, kernel_Delta);
        // get erosion of src using dilation
        cv::erode(dilation, erosion, kernel_Delta);
        // convert the erosion to CV_32FC1
        erosion.convertTo(erosion, CV_32FC1);
        close_ = cv::min(erosion, src_);
        NWTH = src_ - close_;
        cv::max(RW, NWTH, RW);


        cv::erode(src, erosion_b, kernel_Delta);
        cv::dilate(erosion_b, dilation_b, kernel_b);
        dilation_b.convertTo(dilation_b, CV_32FC1);
        open_ = cv::max(dilation_b, src_);
        NBTH = open_ - src_;
        cv::max(RB, NBTH, RB);


        erosion.convertTo(erosion, CV_8UC1);
        dilation_b.convertTo(dilation_b, CV_8UC1);
    }
    
    // get dst
    dst = alpha*src_ + beta*RW - gamma*RB;
    dst.setTo(0, dst < 0);
    dst.setTo(255, dst > 255);
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

cv::Mat IR_Morphology::morphologyCenterOperator(int kernelSize, int threshold){
    cv::Mat src_ = src.clone();
    src_.convertTo(src_, CV_32FC1);
    cv::Mat mc = morphologyCenterOperator_(src, kernelSize);
    cv::Mat mc_ = 255.0 - mc;
    cv::Mat BR1 = cv::max(src_-mc-threshold, 0.0);
    cv::Mat BR2 = cv::max(src_-mc_-threshold, 0.0);
    cv::Mat FBR = cv::max(BR1, BR2);
    FBR.convertTo(FBR, CV_8UC1);
    return FBR;
}


cv::Mat IR_Morphology::morphologyCenterOperator_(cv::Mat src, int kernelSize){
    cv::Mat open;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(src, open, cv::MORPH_OPEN, kernel);
    cv::Mat close;
    cv::morphologyEx(src, close, cv::MORPH_CLOSE, kernel);

    cv::Mat tmp1 = cv::min(open,close);
    cv::Mat tmp2 = cv::max(open,close);
    cv::Mat tmp3 = cv::max(tmp1, src);

    cv::Mat dst = cv::min(
        tmp3,
        tmp2
    );  
    dst.convertTo(dst, CV_32FC1);
    return dst;
}








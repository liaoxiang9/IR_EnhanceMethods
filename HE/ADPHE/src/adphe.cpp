#include "adphe.h"

void adphe::set(const std::string& imagePath){
    src = cv::imread(imagePath, 0);
    if(src.empty()){
        std::cout << "image is empty" << std::endl;
    }
}

void adphe::getHist(){
    float histRange[] ={0, 256};
    const float* ranges[] = {histRange};
    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, ranges,  true, false);
}

float* adphe::getLimit(){
    // get the upper limit of the histogram
    std::vector<float> localMaximum;
    // remove zeros from hist
    std::vector<float> nonZeros;
    int L=0;
    for(int i = 0; i < histSize; i++){
        if(hist.at<float>(i) != 0){
            nonZeros.push_back(hist.at<float>(i));
            L++;
        }
    }
    // overlook the nonZeros
    for(int i=winSize/2; i<L-winSize/2; i++){
        float max = 0;
        for(int j=i-winSize/2; j<i+winSize/2; j++){
            if(nonZeros[j] > max){
                max = nonZeros[j];
            }
        }
        if(nonZeros[i] == max){
            localMaximum.push_back(nonZeros[i]);
        }
    }
    float sum = std::accumulate(localMaximum.begin(), localMaximum.end(), 0.0);
    float upperLimit = sum / localMaximum.size();
    
    // get the lower limit of the histogram
    int Ntotal = src.rows * src.cols;
    float lowerLimit = std::min(upperLimit*L, (float)Ntotal)  / histSize;
    float* limit = new float[4];
    limit[0] = lowerLimit;
    limit[1] = upperLimit;
    limit[2] = L;
    limit[3] = Ntotal;
    return limit;
}

cv::Mat adphe::clipHist(const cv::Mat& hist_, float lowerLimit, float upperLimit){
    cv::Mat result = hist_.clone();
    for(int i=0; i<histSize; i++){
        if(hist_.at<float>(i) >= upperLimit){
            result.at<float>(i) = upperLimit;
        }
        else if(hist_.at<float>(i) <= lowerLimit && hist_.at<float>(i) != 0){
            result.at<float>(i) = lowerLimit;
        }
        else;
    }
    return result;
}

float adphe::getCV(const cv::Mat& mat){
    float mean = cv::mean(mat)[0];
    float sum = 0;
    for(int i=0; i<mat.rows; i++){
        for(int j=0; j<mat.cols; j++){
            sum += pow(mat.at<float>(i, j) - mean, 2);
        }
    }
    float variance = sum / (mat.rows * mat.cols);
    float CV = sqrt(variance) / mean;
    return CV;
}

cv::Mat adphe::mainProcess(){
    // 先求图像的初始直方图
    if(hist.empty()){
        getHist();
    }
    // 求初始直方图的CV
    float CV0 = getCV(hist);
    // 求初始的上下限
    float* limit = getLimit();
    
    float T_up0 = limit[1];
    float T_down0 = limit[0];
    float L = limit[2];
    float Ntotal = limit[3];
    // 截取初始直方图
    cv::Mat h_m = clipHist(hist, T_down0, T_up0);
    // 计算截取直方图的CV
    float CV1 = getCV(h_m);
    // 计算CV的比值
    float R_CV0 = CV1 / CV0;
    // 定义最终的上下限
    float T_up = 0.0;
    float T_down = 0.0;
    float R_CV = 0.0;
    if(R_CV0 < 0.5){
        float R_CV_exp = 0.5 * 0.6 + R_CV0 * 0.4;
        double min, max;
        cv::minMaxLoc(hist, &min, &max);
        float T_up1 = max;
        while(std::abs(T_up0 - T_up1) > 1){
            T_up = 0.5 * (T_up0 + T_up1);
            T_down = std::min(Ntotal, T_up * L) / histSize;
            h_m = clipHist(hist, T_down, T_up);
            CV1 = getCV(h_m);
            R_CV = CV1 / CV0;
            if(R_CV < R_CV_exp){
                T_up0 = T_up;
            }
            else{
                T_up1 = T_up;
            }
        }    
    }else{
        T_up = T_up0;
        T_down = T_down0;
    }
    float mean_h = cv::mean(hist)[0];
    while(T_down > 0.2 * mean_h){
        T_down = mean_h* 0.2;
        h_m = clipHist(hist, T_down, T_up);
    }

    h_m = h_m / cv::sum(h_m)[0];
    cv::Mat h_m_cdf = h_m.clone();
    for(int i=1; i<histSize; i++){
        h_m_cdf.at<float>(i) = h_m_cdf.at<float>(i) + h_m_cdf.at<float>(i-1);
    }

    cv::Mat result = src.clone();
    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            float value = h_m_cdf.at<float>(src.at<uchar>(i, j)) * 255;
            if (value > 255){
                value = 255;
            }
            result.at<uchar>(i, j) = (uchar)value;
        }
    }

    delete[] limit;
    return result;
}




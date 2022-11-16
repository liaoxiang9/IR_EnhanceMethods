#include "agcch.h"

agcch::~agcch()
{
    std::cout << "Destructor called" << std::endl;
}

void agcch::getHist(){
    float histRange[] ={0, 256};
    const float* ranges[] = {histRange};
    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, ranges,  true, false);
    hist = hist / (src.rows * src.cols);
}


// Image enhancement based on globally adaptive gamma correction
cv::Mat agcch::iegagc(){
    cv::Mat dst = src.clone();
    if(hist.empty()){
        getHist();
    }
    cv::Mat histCum = hist.clone();
    for(int i = 1; i < histCum.rows; i++){
        histCum.at<float>(i) = histCum.at<float>(i) + histCum.at<float>(i - 1);
    }

    cv::Mat pdf_gw = cv::Mat::zeros(histCum.rows, 1, CV_32FC1);

    // Calculate the min anx max of the histogram
    double min = 0;
    double max = 0;
    cv::minMaxLoc(hist, &min, &max);
    

    float alpha = 0.0;
    float pdf_l = 0.0;
    for(int i = 0; i < histCum.rows; i++){
        alpha = histCum.at<float>(i);
        pdf_l = hist.at<float>(i);
        pdf_gw.at<float>(i) = pow(((pdf_l - min)/(max - min)), alpha) * max;
    }

    // Calculate the sum of the pdf_gw
    float sum = cv::sum(pdf_gw)[0];
    pdf_gw = pdf_gw / sum;
    cv::Mat cdf_gw = pdf_gw.clone();
    for(int i = 1; i < cdf_gw.rows; i++){
        cdf_gw.at<float>(i) = cdf_gw.at<float>(i) + cdf_gw.at<float>(i - 1);
        // std::cout << "cdf_gw: " <<i<<":\t"<< cdf_gw.at<float>(i) << std::endl;
    }
    // padding according to the window size
    int pad = winSize / 2;
    cv::Mat padSrc = cv::Mat::zeros(src.rows + 2 * pad, src.cols + 2 * pad, CV_8UC1);
    cv::copyMakeBorder(src, padSrc, pad, pad, pad, pad, cv::BORDER_CONSTANT, 0);
    cv::Mat win;
    for(int i=pad; i < padSrc.rows - pad; i++){
        for(int j = pad; j < padSrc.cols - pad; j++){
            float val = (float)padSrc.at<uchar>(i, j);
            // calculate the mean of the window
            win = padSrc(cv::Rect(j - pad, i - pad, winSize, winSize)); 
            float mean = cv::mean(win)[0];
            dst.at<uchar>(i - pad, j - pad) = val * pow((val / mean), (1 - cdf_gw.at<float>(val)));
            // std::cout <<"src："<<val<<"bottom:"<< (val / mean)  << "dst: " << (int)dst.at<uchar>(i - pad, j - pad) << " gamma:" <<1 - cdf_gw.at<float>(val)<<std::endl;
            // std::cout << "gamma: " << 1 - cdf_gw.at<float>(val) << std::endl;
        }
    }
    return dst;
    }

cv::Mat agcch::ielagc(){
    cv::Mat dst = src.clone();
    if(hist.empty()){
        getHist();
    }

    // cv::Mat ks = cv::Mat::zeros(histSize, 1, CV_32FC1);
    float omega0 = 0.0;
    float omega1 = 0.0;
    float mu0 = 0.0;
    float mu1 = 0.0;
    float mu = 0.0;
    int k_opt = 0;
    float max = 0.0;
    float var = 0.0;
    for(int k=0;k<histSize;k++){
        // 计算前k个hist值的和
        for(int i=0;i<k;i++){
            omega0 += hist.at<float>(i);
        }
        // 计算后k个hist值的和
        for(int j=k;j<histSize;j++){
            omega1 += hist.at<float>(j);
        }

        // 计算前k个hist值的均值
        for(int i=0;i<k;i++){
            mu0 += i * hist.at<float>(i);
        }
        mu0 = mu0 / omega0;

        // 计算后k个hist值的均值
        for(int j=k;j<histSize;j++){
            mu1 += j * hist.at<float>(j);
        }
        mu1 = mu1 / omega1;

        // calculate the mean of the the whole image
        mu = mu0 * omega0 + mu1 * omega1;
        var = omega0 * pow((mu0 - mu), 2) + omega1 * pow((mu1 - mu), 2);
        // ks.at<float>(k) = var;
        if(var > max){
            max = var;
            k_opt = k;
        }
    }

    int k2 = k_opt;

    // calculate the k1 and k3
    int k1 = 0;
    int k3 = 0;
    mu0 = 0.0;
    mu1 = 0.0;
    omega0 = 0.0;
    omega1 = 0.0;
    max = 0.0;
    for(int k=0;k<k2;k++){
        // 计算前k个hist值的和
        for(int i=0;i<k;i++){
            omega0 += hist.at<float>(i);
        }
        // 计算后k个hist值的和
        for(int j=k;j<k2;j++){
            omega1 += hist.at<float>(j);
        }

        // 计算前k个hist值的均值
        for(int i=0;i<k;i++){
            mu0 += i * hist.at<float>(i);
        }
        mu0 = mu0 / omega0;

        // 计算后k个hist值的均值
        for(int j=k;j<k2;j++){
            mu1 += j * hist.at<float>(j);
        }
        mu1 = mu1 / omega1;

        // calculate the mean of the the whole image
        mu = mu0 * omega0 + mu1 * omega1;
        var = omega0 * pow((mu0 - mu), 2) + omega1 * pow((mu1 - mu), 2);
        // ks.at<float>(k) = var;
        if(var > max){
            max = var;
            k1 = k;
        }
    }

    mu0 = 0.0;
    mu1 = 0.0;
    omega0 = 0.0;
    omega1 = 0.0;
    max = 0.0;
    for(int k=k2+1;k<histSize;k++){
        // 计算前k个hist值的和
        for(int i=k2+1;i<k;i++){
            omega0 += hist.at<float>(i);
        }
        // 计算后k个hist值的和
        for(int j=k;j<histSize;j++){
            omega1 += hist.at<float>(j);
        }

        // 计算前k个hist值的均值
        for(int i=k2+1;i<k;i++){
            mu0 += i * hist.at<float>(i);
        }
        mu0 = mu0 / omega0;

        // 计算后k个hist值的均值
        for(int j=k;j<histSize;j++){
            mu1 += j * hist.at<float>(j);
        }
        mu1 = mu1 / omega1;

        // calculate the mean of the the whole image
        mu = mu0 * omega0 + mu1 * omega1;
        var = omega0 * pow((mu0 - mu), 2) + omega1 * pow((mu1 - mu), 2);
        // ks.at<float>(k) = var;
        if(var > max){
            max = var;
            k2 = k;
        }
    }

    int k0 = 0;
    int k4 = histSize - 1;

    // calculate the alpha0, alpha1, alpha2, alpha3
    float alpha0 = hist.at<float>(k0) + hist.at<float>(k1);
    float alpha1 = hist.at<float>(k1) + hist.at<float>(k2);
    float alpha2 = hist.at<float>(k2) + hist.at<float>(k3);
    float alpha3 = hist.at<float>(k3) + hist.at<float>(k4);


    

    cv::Mat histCum = hist.clone();
    for(int i = 1; i < histCum.rows; i++){
        histCum.at<float>(i) = histCum.at<float>(i) + histCum.at<float>(i - 1);
    }

    cv::Mat pdf_gw = cv::Mat::zeros(histCum.rows, 1, CV_32FC1);

    // Calculate the min anx max of the histogram
    double min = 0;
    double max_ = 0;
    cv::minMaxLoc(hist, &min, &max_);
    

    float alpha = 0.0;
    float pdf_l = 0.0;
    for(int i = 0; i < histCum.rows; i++){
        pdf_l = hist.at<float>(i);
        if(i<k1){
            alpha = alpha0;
            }else if(i<k2){
            alpha = alpha1;
            }else if(i<k3){
            alpha = alpha2;
            }else{
            alpha = alpha3;
        }
        pdf_gw.at<float>(i) = pow(((pdf_l - min)/(max_ - min)), alpha) * max_;
    }

    // Calculate the sum of the pdf_gw
    float sum = cv::sum(pdf_gw)[0];
    pdf_gw = pdf_gw / sum;
    cv::Mat cdf_gw = pdf_gw.clone();
    for(int i = 1; i < cdf_gw.rows; i++){
        cdf_gw.at<float>(i) = cdf_gw.at<float>(i) + cdf_gw.at<float>(i - 1);
        // std::cout << "cdf_gw: " <<i<<":\t"<< cdf_gw.at<float>(i) << std::endl;
    }
    // padding according to the window size
    int pad = winSize / 2;
    cv::Mat padSrc = cv::Mat::zeros(src.rows + 2 * pad, src.cols + 2 * pad, CV_8UC1);
    // 镜像填充
    cv::copyMakeBorder(src, padSrc, pad, pad, pad, pad, cv::BORDER_REFLECT);
    // cv::copyMakeBorder(src, padSrc, pad, pad, pad, pad, cv::);
    cv::Mat win;
    for(int i=pad; i < padSrc.rows - pad; i++){
        for(int j = pad; j < padSrc.cols - pad; j++){
            float val = (float)padSrc.at<uchar>(i, j);
            // calculate the mean of the window
            win = padSrc(cv::Rect(j - pad, i - pad, winSize, winSize)); 
            float mean = cv::mean(win)[0];
            dst.at<uchar>(i - pad, j - pad) = val * pow((val / mean), (1 - cdf_gw.at<float>(val)));
            // std::cout <<"src："<<val<<"bottom:"<< (val / mean)  << "dst: " << (int)dst.at<uchar>(i - pad, j - pad) << " gamma:" <<1 - cdf_gw.at<float>(val)<<std::endl;
            // std::cout << "gamma: " << 1 - cdf_gw.at<float>(val) << std::endl;
        }
    }
    return dst;
    
}



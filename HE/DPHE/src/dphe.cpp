#include "dphe.h"

cv::Mat dphe::get_result()
{
    cv::Mat accumulate_hist = get_accumulate_hist();
        cv::Mat result = src.clone();
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                result.at<uchar>(i, j) = (uchar)accumulate_hist.at<float>(src.at<uchar>(i, j));
            }
        }
        return result;
}

void dphe::setLowLimit()
{
    cv::Scalar s = cv::sum(src);
    lowLimit = cv::mean(src)[0] / s[0];
}

cv::Mat dphe::clip_hist(){
    cv::Mat hist = getHist();
    cv::Mat clipped_hist = hist.clone();
    setThreshhold();
    setLowLimit();
    for(int i=0;i<256;i++){
        if(clipped_hist.at<float>(i)>0.2*lowLimit && clipped_hist.at<float>(i)<lowLimit){
            clipped_hist.at<float>(i) = lowLimit;
        }
    }
    clipped_hist.setTo(0, clipped_hist <= 0.2*lowLimit);
    clipped_hist.setTo(threshhold, clipped_hist > threshhold);
    return clipped_hist;
}

cv::Mat dphe::get_accumulate_hist() {
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

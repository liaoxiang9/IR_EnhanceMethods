#include "rangeCompress.h"

cv::Mat rangeCompress::histogramCompress(cv::Mat &src, float threshPercent){
    // judge the depth of the src is 16bit
    if(src.depth()!=CV_16U){
        std::cout<<"The depth of the src is not 16bit"<<std::endl;
        return src;
    }
    
    // get the histogram of the src
    cv::Mat hist;
    int histSize = 65536;
    float range[] = { 0, 65536 } ;
    const float* histRange[] = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::calcHist( &src, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate );
    // get the total number of the image
    int totalNum=src.rows*src.cols;
    int threshold=totalNum*threshPercent;
    // gei binart hist
    cv::Mat binaryHist = cv::Mat::zeros(hist.size(), hist.type());
    int n_valid=0;    
    for(int i=0;i<hist.rows;i++){
        if(hist.at<float>(i)>=threshold){
            binaryHist.at<float>(i)=1;
            n_valid++;
        }else{
            binaryHist.at<float>(i)=0;
        }
    }
    // get cumulative hist
    cv::Mat cumulativeHist = cv::Mat::zeros(hist.size(), hist.type());
    int outputBitWidth=255;
    int R = std::min(n_valid, outputBitWidth);
    std::cout << "R: " << R << std::endl;
    cumulativeHist.at<float>(0)=0.0;
    for(int i=1;i<hist.rows;i++){
        cumulativeHist.at<float>(i)=cumulativeHist.at<float>(i-1)+binaryHist.at<float>(i);
    }
    cumulativeHist=cumulativeHist/n_valid;
    // get the mapping table
    cv::Mat mappingTable = cv::Mat::zeros(hist.size(), hist.type());
    for(int i=0;i<hist.rows;i++){
        mappingTable.at<float>(i)=cumulativeHist.at<float>(i)*R;
    }
    // get the compressed 8bit image
    cv::Mat compressedImage = cv::Mat::zeros(src.size(), CV_8U);
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            compressedImage.at<uchar>(i,j)=mappingTable.at<float>(src.at<ushort>(i,j));
        }
    }
    return compressedImage;
}

cv::Mat rangeCompress::linearCompress(cv::Mat& src){
    // judge the depth of the src is 16bit
    if(src.depth()!=CV_16U){
        std::cout<<"The depth of the src is not 16bit"<<std::endl;
        return src;
    }
    // get the min and max value of the src
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);
    // get the mapping table
    cv::Mat result = cv::Mat::zeros(src.size(), CV_8U);
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            result.at<uchar>(i,j)=(src.at<ushort>(i,j)-minVal)/(maxVal-minVal)*255;
        }
    }
    return result;
}

cv::Mat rangeCompress::gammaCurveCompress(cv::Mat& src, float gamma){
    // judge the depth of the src is 16bit
    if(src.depth()!=CV_16U){
        std::cout<<"The depth of the src is not 16bit"<<std::endl;
        return src;
    }
    // get the min and max value of the src
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);
    std::cout<<"minVal: "<<minVal<<std::endl;
    std::cout<<"maxVal: "<<maxVal<<std::endl;
    // get the mapping table
    cv::Mat result = cv::Mat::zeros(src.size(), CV_8U);
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            result.at<uchar>(i,j)=pow((src.at<ushort>(i,j)-minVal)/(maxVal-minVal), gamma)*255;
        }
    }

    // find the gamma to make the entropy of the image to be biggest
    
    return result;
}

float rangeCompress::getEntropy(cv::Mat& src){
    // get the histogram of the src
    cv::Mat hist;
    int histSize = 65536;
    float range[] = { 0, 65536 } ;
    const float* histRange[] = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::calcHist( &src, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate );
    // get the total number of the image
    int totalNum=src.rows*src.cols;
    // get the entropy
    float entropy=0.0;
    for(int i=0;i<hist.rows;i++){
        if(hist.at<float>(i)!=0){
            entropy+=hist.at<float>(i)/totalNum*log2(hist.at<float>(i)/totalNum);
        }
    }
    return -entropy;
}
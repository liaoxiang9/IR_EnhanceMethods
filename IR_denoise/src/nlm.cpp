#include "nlm.h"
#include <cmath>


std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
    }





void NLM::denoise(cv::Mat &dst) const
{
    dst = cv::Mat::zeros(src.size(), CV_32FC1);
    int half_template_size = template_size / 2;
    int half_search_size = search_size / 2;
    int pad_size = half_template_size + half_search_size;
    // pad for the src
    cv::Mat src_pad;
    cv::copyMakeBorder(src, src_pad, pad_size, pad_size, pad_size, pad_size, cv::BORDER_CONSTANT, 0);
    src_pad.convertTo(src_pad, CV_32FC1);
    cv::Mat patch;
    // initialize the weight
    cv::Mat weight = cv::Mat::zeros(cv::Size(search_size, search_size), CV_32FC1);
    // initialize the weight_sum
    // float weight_sum = 0.0;
    float sum = 0.0;
    cv::Mat template_img;
    cv::Mat search_area;
    // floop for the src_pad
    for (int i = pad_size; i < src_pad.rows - pad_size; i++)
    {
        for (int j = pad_size; j < src_pad.cols - pad_size; j++)
        {
            sum = 0.0;
            // get the template
            template_img = src_pad(cv::Rect(j - half_template_size, i - half_template_size, template_size, template_size));
            // get the search area
            search_area = src_pad(cv::Rect(j - half_search_size, i - half_search_size, search_size, search_size));
            // floop for the search_area
            for (int m = 0; m < search_size; m++)
            {
                for (int n = 0; n < search_size; n++)
                {
                    // get the patch center at (m,n)
                    patch = src_pad(cv::Rect(
                        j - half_search_size + n - half_template_size,
                        i - half_search_size + m - half_template_size,
                        template_size, template_size));
                    // calculate the distance between template and patch
                    float tmp = exp(-1 * distance(template_img, patch) / (h*h));
                    weight.at<float>(m, n) =  tmp;
                    sum += tmp;
                }
            }
            // weight_sum = cv::sum(weight)[0];
            // std::cout << "weight_sum: " << weight_sum << std::endl;
            dst.at<float>(i - pad_size, j - pad_size) = cv::sum(weight.mul(search_area))[0] / sum;
        }
    }
    dst.setTo(255, dst > 255);
    dst.convertTo(dst, CV_8UC1);
}

float NLM::distance(const cv::Mat &template_img, const cv::Mat &patch) const
{
    // get the template and patch size
    int template_size = template_img.rows;
    cv::Mat MSE;
    cv::pow(template_img - patch, 2, MSE);
    float dist = cv::sum(MSE)[0] / (template_size * template_size);
    return dist;
}
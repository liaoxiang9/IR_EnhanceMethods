#include "phe.h"

void phe::set(const std::string& img_path)
{
    src_img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (src_img.empty()) {
        throw std::runtime_error("Can't open image");
    }
}
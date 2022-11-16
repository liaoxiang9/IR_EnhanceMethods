#include "IR_SaliencyExtraction.h"

int main(){
    IR_SaliencyExtraction ir_saliencyExtraction("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    // cv::Mat saliencyMap = ir_saliencyExtraction.frequencyTunedSaliencyMap(ir_saliencyExtraction.src, 1.5 , 0.1, 7, 1.2);
    cv::Mat src = ir_saliencyExtraction.getSrc();
    // cv::Mat dst = ir_saliencyExtraction.multiScaleFTSaliencyMap(src=src, r=3, t0=1, delta_t=2, alpha1=0.5, alpha2=0.5, sigma1=1.2, sigma2=1.2*1.6, n=9);
    cv::Mat dst = ir_saliencyExtraction.multiScaleFTSaliencyMap(src, 3, 1, 2, 0.5, 0.5, 1.5, 1.5*1.6, 9);
    cv::imwrite("../result/multiSacleFT.jpg", dst);
    return 0;
}
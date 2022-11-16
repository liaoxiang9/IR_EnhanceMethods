#include "IR_Morphology.h"

int main(){
    IR_Morphology im("/home/lx/IR_Enhance_Methods/src_images/1.jpg");
    // cv::Mat dst = im.modifiedTopHat(55, 0.5, blackTopHatType);
    // cv::Mat dst = im.modifiedTopHat(55, 0.8, 0.9, 1.5, 2.2);
    // // cv::imshow("dst", dst);
    // // cv::waitKey(0);
    // cv::imwrite("/home/lx/IR_Enhance_Methods/MorphologyBasedMethods/result/modifiedTopHat.jpg", dst);
    // cv::imwrite("../result/modifiedWhiteTopHat", dst2);
    // return 0;

    // cv::Mat dst = im.multiScaleTopHat(5, 9, 5, 2, 1, 4, 1.5, 11);
    // cv::imwrite("../result/multiScaleTopHat.jpg", dst);
    cv::Mat dst  = im.morphologyCenterOperator(15, 5);
    cv::imwrite("../result/morphologyCenterOperator.jpg", dst);
}
#include "ir_vis_fush.h"
#include <iostream>
// 测量时间
#include <chrono>
using namespace std;

int main(){
    string ir_path = "../src/ir.png";
    string vis_path = "../src/vis.png";
    ir_vis_fuse ir_vis_fuse(ir_path, vis_path);
    // 测量时间
    auto start = chrono::steady_clock::now();
    cv::Mat ir_vis_fuse_result = ir_vis_fuse.ir_vis_fuse_process(0.5, 130, 255);
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "ir_vis_fuse_process() takes " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " seconds" << endl;
    cv::imwrite("../result/ir_vis_fuse_result.png", ir_vis_fuse_result);
    return 0;
}
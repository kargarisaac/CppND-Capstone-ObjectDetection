//
// Created by isaac on 5/12/20.
//

#ifndef HELLOWORLD_YOLOV3_H
#define HELLOWORLD_YOLOV3_H

//#include "opencv4/opencv2/opencv.hpp"
//#include "opencv4/opencv2/dnn.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace cv;
using namespace cv::dnn;

class YoloV3 {
public:
    YoloV3(string &yolo_cfg, string &yolo_model);
    void setInputImg(Mat &inputImg);
    void getOutputs(vector<Mat> &outputs);

private:
    Net _net;
    vector<string> _outputNames;
};


#endif //HELLOWORLD_YOLOV3_H

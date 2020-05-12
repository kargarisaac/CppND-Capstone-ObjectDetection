//
// Created by isaac on 5/12/20.
//

#include "YoloV3.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

YoloV3::YoloV3(string &yolo_cfg, string &yolo_model)
{
    _net = readNetFromDarknet(yolo_cfg, yolo_model);
    _net.setPreferableBackend(DNN_BACKEND_OPENCV);
    _net.setPreferableTarget(DNN_TARGET_CPU);
    _outputNames = _net.getUnconnectedOutLayersNames();
}

void YoloV3::setInputImg(Mat &inputImg)
{
    _net.setInput(inputImg);
}

void YoloV3::getOutputs(vector<Mat> &outputs)
{
    _net.forward(outputs, _outputNames);
}

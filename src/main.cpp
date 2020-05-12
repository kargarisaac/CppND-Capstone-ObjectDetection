//
// Created by isaac on 5/12/20.
//

#include <iostream>
#include <string>
#include "YoloV3.h"

string yolo_cfg = "yolov3-tiny-coco/yolov3-tiny.cfg";
string yolo_model = "yolov3-tiny-coco/yolov3-tiny.weights";
string classes_yolov3 = "yolov3-tiny-coco/object_detection_classes_yolov3.txt";

int main(int argc, char** argv)
{
    // Get class name vector
    vector<string> classNamesVec;
    ifstream classNamesFile(classes_yolov3);
    if (classNamesFile.is_open())
    {
        string className;
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    // read the input image and resize it, and the object name we want to detect
    string imageFile;
    string objectName;
    if (argc > 1)
    {
        imageFile = argv[1];
        if (argc > 2) {
            objectName = argv[2];
        } else
        {
            objectName = "car";
        }
    } else
    {
        imageFile = "img.jpg";
        objectName = "car";
    }


    Mat frame = imread(imageFile);
    Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels

    // create net object
    YoloV3 net(yolo_cfg, yolo_model);
    // set input
    net.setInputImg(inputBlob);
    // get outputs
    vector<Mat> outputs;
    net.getOutputs(outputs);

    // get bboxes and class IDs and confidence values
    vector<Rect> boxes;
    vector<int> classIds;
    vector<float> confidences;

    for(auto &output: outputs)
    {
        // Network produces output blob with a shape NxC where N is the number of
        // detected objects and C is the number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        auto* data = (float*)output.data;
        for (int j = 0; j < output.rows; j++, data += output.cols)
        {
            Mat scores = output.row(j).colRange(5, output.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint); //Finds the maximum in scores.
            if (confidence > 0.5)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.emplace_back(Rect(left, top, width, height));
            }
        }
    }

    // draw box for detected objects
    vector<int> indices;

    NMSBoxes(boxes, confidences, 0.5, 0.2, indices); // Perform non maximum suppression given boxes and corresponding scores
    for (auto idx: indices)
    {
        Rect box = boxes[idx];
        string className = classNamesVec[classIds[idx]];
        if (className.compare(objectName)) // check the detected object name
            continue;
        putText(frame, className, box.tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0), 2, 2);
        rectangle(frame, box, Scalar(0, 255, 255), 2, 2, 0);
    }

    // Show all the cars
    imshow("Car Detector Result", frame);
//    imwrite("detected_img2.jpg", frame );
    waitKey(0);
}
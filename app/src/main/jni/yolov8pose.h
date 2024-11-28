//
// Created by wangke on 2024/4/21.
//

#ifndef NCNN_ANDROID_YOLOV8_POSE_YOLOV8POSE_H
#define NCNN_ANDROID_YOLOV8_POSE_YOLOV8POSE_H

#include <vector>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>

#include <net.h>


struct Pose
{
	cv::Rect_<float> rect;
	int label;
	float prob;
	std::vector<float> kps;
};


class Inference
{
public:
    Inference();
    int loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu = false);
    std::vector<Pose> runInference(const cv::Mat &input);
    int draw(cv::Mat& rgb, const std::vector<Pose>& objects);

private:
    ncnn::Net net;

    std::string modelPath{};
    bool gpuEnabled{};

    int target_size;

    // float modelScoreThreshold      {0.45};
    // float modelNMSThreshold        {0.50};


    float meanVals[3];
    float normVals[3];

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //NCNN_ANDROID_YOLOV8_POSE_YOLOV8POSE_H

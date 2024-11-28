#include "yolov8pose.h"
#include <cpu.h>
#include <iostream>
#include <vector>


const std::vector<std::vector<unsigned int>> KPS_COLORS =
        { {0,   255, 0}, {0,   255, 0},  {0,   255, 0}, {0,   255, 0},
          {0,   255, 0},  {255, 128, 0},  {255, 128, 0}, {255, 128, 0},
          {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51,  153, 255},
          {51,  153, 255},{51,  153, 255},{51,  153, 255},{51,  153, 255},
          {51,  153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON =
        { {16, 14},  {14, 12},  {17, 15},  {15, 13},   {12, 13}, {6,  12},
          {7,  13},  {6,  7},   {6,  8},   {7,  9},   {8,  10},  {9,  11},
          {2,  3}, {1,  2},  {1,  3},  {2,  4},  {3,  5},   {4,  6},  {5,  7} };

const std::vector<std::vector<unsigned int>> LIMB_COLORS =
        { {51,  153, 255}, {51,  153, 255},   {51,  153, 255},
          {51,  153, 255}, {255, 51,  255},   {255, 51,  255},
          {255, 51,  255}, {255, 128, 0},     {255, 128, 0},
          {255, 128, 0},   {255, 128, 0},     {255, 128, 0},
          {0,   255, 0},   {0,   255, 0},     {0,   255, 0},
          {0,   255, 0},   {0,   255, 0},     {0,   255, 0},
          {0,   255, 0} };


// sigmoid(x) = 1 / (1 + exp(-x))
// static float sigmoid(const float in)
// {
// 	return 1.f / (1.f + expf(-1.f * in));
// }

// 快速指数
static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float softmax(const float* src, float* dst, int length)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++)
    {
        float score = src[c];
        if (score > alpha)
        {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);  // 使用 fast_exp 代替 expf
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}

static void generate_proposals(
        int stride,
        const ncnn::Mat& feat_blob,
        const float prob_threshold,
        std::vector<Pose>& objects
)
{
    const int reg_max = 16;
    float dst[16];
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;
    const int kps_num = 17;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const float* matat = feat_blob.channel(i).row(j);

            float score = matat[0];
            score = sigmoid(score);
            if (score < prob_threshold)
            {
                continue;
            }

            float x0 = j + 0.5f - softmax(matat + 1, dst, 16);
            float y0 = i + 0.5f - softmax(matat + (1 + 16), dst, 16);
            float x1 = j + 0.5f + softmax(matat + (1 + 2 * 16), dst, 16);
            float y1 = i + 0.5f + softmax(matat + (1 + 3 * 16), dst, 16);

            x0 *= stride;
            y0 *= stride;
            x1 *= stride;
            y1 *= stride;

            std::vector<float> kps;
            for(int k=0; k<kps_num; k++)
            {
                float kps_x = (matat[1 + 64 + k * 3] * 2.f+ j) * stride;
                float kps_y = (matat[1 + 64 + k * 3 + 1] * 2.f + i) * stride;
                float kps_s = sigmoid(matat[1 + 64 + k * 3 + 2]);

                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }

            Pose obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = 0;
            obj.prob = score;
            obj.kps = kps;
            objects.push_back(obj);
        }
    }

}

static float clamp(
        float val,
        float min = 0.f,
        float max = 1280.f
)
{
    return val > min ? (val < max ? val : max) : min;
}

// 计算 IoU (Intersection over Union) 的函数
static float compute_iou(const cv::Rect& box1, const cv::Rect& box2)
{
    float inter_area = (box1 & box2).area();
    float union_area = box1.area() + box2.area() - inter_area;
    return inter_area / union_area;
}


static void non_max_suppression(
        std::vector<Pose>& proposals,
        std::vector<Pose>& results,
        int orin_h,
        int orin_w,
        float dh = 0,
        float dw = 0,
        float ratio_h = 1.0f,
        float ratio_w = 1.0f,
        float conf_thres = 0.25f,
        float iou_thres = 0.65f
)
{
    results.clear();
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    // 1. 将 proposals 中的矩形框和分数提取出来
    for (auto& pro : proposals)
    {
        bboxes.push_back(pro.rect);
        scores.push_back(pro.prob);
        labels.push_back(pro.label);
        kpss.push_back(pro.kps);
    }

    // 2. 按照分数进行排序，排序后从高到低进行 NMS
    std::vector<int> sorted_indices(scores.size());
    for (int i = 0; i < scores.size(); ++i)
    {
        sorted_indices[i] = i;
    }

    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });

    // 3. 执行非最大抑制
    std::vector<bool> keep(scores.size(), true);
    for (int i = 0; i < sorted_indices.size(); ++i)
    {
        if (!keep[sorted_indices[i]]) continue;

        const auto& box_i = bboxes[sorted_indices[i]];
        float score_i = scores[sorted_indices[i]];

        if (score_i < conf_thres)
            break;

        for (int j = i + 1; j < sorted_indices.size(); ++j)
        {
            if (!keep[sorted_indices[j]]) continue;

            const auto& box_j = bboxes[sorted_indices[j]];
            float iou = compute_iou(box_i, box_j);

            // 如果 IOU 大于阈值，则将 box_j 去除
            if (iou > iou_thres)
            {
                keep[sorted_indices[j]] = false;
            }
        }
    }

    // 4. 将满足条件的框添加到结果中
    for (int i = 0; i < sorted_indices.size(); ++i)
    {
        if (keep[sorted_indices[i]])
        {
            const auto& bbox = bboxes[sorted_indices[i]];
            float score = scores[sorted_indices[i]];
            int label = labels[sorted_indices[i]];
            std::vector<float> kps = kpss[sorted_indices[i]];

            float x0 = bbox.x;
            float y0 = bbox.y;
            float x1 = bbox.x + bbox.width;
            float y1 = bbox.y + bbox.height;

            x0 = (x0 - dw) / ratio_w;
            y0 = (y0 - dh) / ratio_h;
            x1 = (x1 - dw) / ratio_w;
            y1 = (y1 - dh) / ratio_h;

            x0 = clamp(x0, 0.f, orin_w);
            y0 = clamp(y0, 0.f, orin_h);
            x1 = clamp(x1, 0.f, orin_w);
            y1 = clamp(y1, 0.f, orin_h);

            Pose obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.prob = score;
            obj.label = label;
            obj.kps = kps;

            for (int n = 0; n < obj.kps.size(); n += 3)
            {
                obj.kps[n] = clamp((obj.kps[n] - dw) / ratio_w, 0.f, orin_w);
                obj.kps[n + 1] = clamp((obj.kps[n + 1] - dh) / ratio_h, 0.f, orin_h);
            }

            results.push_back(obj);
        }
    }
}

Inference::Inference(){
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Inference::loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu)
{
    target_size = modelInputShape;
    gpuEnabled = useGpu;

    net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();

#if NCNN_VULKAN
    net.opt.use_vulkan_compute = useGpu;
#endif

    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blob_pool_allocator;
    net.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "yolov8%s-pose_ncnn_model/model.ncnn.param", modeltype);
    sprintf(modelpath, "yolov8%s-pose_ncnn_model/model.ncnn.bin", modeltype);

    net.load_param(mgr, parampath);
    net.load_model(mgr, modelpath);

    this->meanVals[0] = meanVals[0];
    this->meanVals[1] = meanVals[1];
    this->meanVals[2] = meanVals[2];
    this->normVals[0] = normVals[0];
    this->normVals[1] = normVals[1];
    this->normVals[2] = normVals[2];
    return 0;
}

std::vector<Pose> Inference::runInference(const cv::Mat &bgr)
{
//    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int w = img_w;
    int h = img_h;

    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = (int)(h * scale);
    }
    else {
        scale = (float)target_size / h;
        h = target_size;
        w = (int)(w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w + 32 - 1) / 32 * 32 - w;
    int hpad = (h + 32 - 1) / 32 * 32 - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,  wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(meanVals, normVals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("images", in_pad);


    std::vector<Pose> proposals;

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output0", out);

        std::vector<Pose> objects8;
        generate_proposals(8, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("321", out);

        std::vector<Pose> objects16;
        generate_proposals(16, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("340", out);

        std::vector<Pose> objects32;
        generate_proposals(32, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    std::vector<Pose> poses;
    non_max_suppression(proposals, poses,
                        img_h, img_w, hpad / 2, wpad / 2,
                        scale, scale, prob_threshold, nms_threshold);


    return poses;

}

int Inference::draw(cv::Mat& rgb, const std::vector<Pose>& objects) {

    cv::Mat res = rgb;
    for (auto& obj : objects) {
        cv::rectangle(res, obj.rect, { 0, 0, 255 }, 2);

        char text[256];
        sprintf(text, "person %.1f%%", obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x1 = obj.rect.x;
        int y1 = obj.rect.y - label_size.height - baseLine;
        if (y1 < 0)
            y1 = 0;
        if (x1 + label_size.width > rgb.cols)
            x1 = rgb.cols - label_size.width;
        cv::putText(rgb, text, cv::Point(x1, y1 + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        auto& kps = obj.kps;
        for (int k = 0; k < 17 + 2; k++) {
            if (k < 17) {
                int kps_x = (int)std::round(kps[k * 3]);
                int kps_y = (int)std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.4f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, { kps_x, kps_y }, 5, kps_color, -1);
                }
            }
            auto& ske = SKELETON[k];
            int pos1_x = (int)std::round(kps[(ske[0] - 1) * 3]);
            int pos1_y = (int)std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = (int)std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = (int)std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, { pos1_x, pos1_y }, { pos2_x, pos2_y }, limb_color, 2);
            }
        }
    }
    return 0;
}

/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2024, AXERA Semiconductor Co., Ltd. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

/*
 * Note: For the YOLO11 series exported by the ultralytics project.
 * Author: LittleMouse
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "base/common.hpp"
#include "base/detection.hpp"

#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include <axcl.h>
#include "ax_model_runner/ax_model_runner_axcl.hpp"

const int DEFAULT_IMG_H = 640;
const int DEFAULT_IMG_W = 640;
const int WARMUP_COUNT = 5;

const char *DEFAULT_MODEL_FILE = R"(D:\yolo26\yolo26m.axmodel)";
const char *DEFAULT_INPUT_DIR = R"(D:\yolo26\images)";
const char *DEFAULT_OUTPUT_DIR_NAME = "output";

const char *CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

int NUM_CLASS = 80;

const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.45f;

namespace fs = std::filesystem;

namespace
{
    using Clock = std::chrono::steady_clock;

    struct RecognitionTiming
    {
        double image_load_ms = 0.0;
        double preprocess_ms = 0.0;
        double input_copy_ms = 0.0;
        double inference_call_ms = 0.0;
        double host_to_device_ms = 0.0;
        double model_inference_ms = 0.0;
        double device_to_host_ms = 0.0;
        double postprocess_ms = 0.0;
        double recognition_pipeline_ms = 0.0;
        double input_to_result_ms = 0.0;
    };

    double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    void print_recognition_timing(const RecognitionTiming &timing, bool success)
    {
        fprintf(stdout, "识别耗时明细（AXCL 子项已包含在 AXCL 调用总耗时中）：\n");
        fprintf(stdout, "  图片读取/解码：%8.3f ms\n", timing.image_load_ms);
        fprintf(stdout, "  图像预处理：   %8.3f ms\n", timing.preprocess_ms);
        fprintf(stdout, "  输入缓冲区复制：%8.3f ms\n", timing.input_copy_ms);
        fprintf(stdout, "  AXCL 调用总耗时：%8.3f ms\n", timing.inference_call_ms);
        fprintf(stdout, "    主机到设备（H2D）：%8.3f ms\n", timing.host_to_device_ms);
        fprintf(stdout, "    模型推理：         %8.3f ms\n", timing.model_inference_ms);
        fprintf(stdout, "    设备到主机（D2H）：%8.3f ms\n", timing.device_to_host_ms);
        fprintf(stdout, "  后处理：       %8.3f ms\n", timing.postprocess_ms);
        fprintf(stdout, "识别流程耗时（不含图片读取和输出处理）：%.3f ms\n", timing.recognition_pipeline_ms);
        fprintf(stdout, "输入到结果耗时（不含绘制和保存）：%.3f ms\n", timing.input_to_result_ms);
        fprintf(stdout, "识别状态：%s\n", success ? "成功" : "失败");
    }

    void print_output_timing(const detection::DrawObjectsTiming &timing, bool success)
    {
        fprintf(stdout, "输出处理耗时（不计入识别耗时）：\n");
        fprintf(stdout, "  结果绘制：%8.3f ms\n", timing.render_ms);
        fprintf(stdout, "  结果保存：%8.3f ms\n", timing.save_ms);
        fprintf(stdout, "输出状态：%s\n", success ? "成功" : "失败");
    }

    template<typename Timing>
    void print_timing_statistics(const char *name, const std::vector<Timing> &timings, double Timing::*field)
    {
        double total = 0.0;
        double minimum = timings.front().*field;
        double maximum = minimum;
        for (const auto &timing : timings)
        {
            const double value = timing.*field;
            total += value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }

        fprintf(stdout, "  %s：平均 %8.3f ms，最小 %8.3f ms，最大 %8.3f ms\n", name,
                total / static_cast<double>(timings.size()), minimum, maximum);
    }

    bool is_supported_image_file(const fs::path &file)
    {
        static const std::array<std::string, 7> SUPPORTED_EXTENSIONS = {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"};

        auto extension = file.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value)
        {
            return static_cast<char>(std::tolower(value));
        });

        return std::find(SUPPORTED_EXTENSIONS.begin(), SUPPORTED_EXTENSIONS.end(), extension) != SUPPORTED_EXTENSIONS.end();
    }

    bool collect_image_files(const fs::path &input_dir, std::vector<fs::path> &image_files)
    {
        std::error_code error;
        fs::directory_iterator iterator(input_dir, error);
        if (error)
        {
            fprintf(stderr, "Open input directory failed: %s (%s)\n", input_dir.string().c_str(), error.message().c_str());
            return false;
        }

        const fs::directory_iterator end;
        while (iterator != end)
        {
            std::error_code type_error;
            if (iterator->is_regular_file(type_error) && is_supported_image_file(iterator->path()))
            {
                image_files.push_back(iterator->path());
            }
            else if (type_error)
            {
                fprintf(stderr, "Read directory entry failed: %s (%s)\n", iterator->path().string().c_str(), type_error.message().c_str());
            }

            iterator.increment(error);
            if (error)
            {
                fprintf(stderr, "Scan input directory failed: %s (%s)\n", input_dir.string().c_str(), error.message().c_str());
                return false;
            }
        }

        std::sort(image_files.begin(), image_files.end(), [](const fs::path &left, const fs::path &right)
        {
            return left.filename().native() < right.filename().native();
        });
        return true;
    }

    bool load_image(const fs::path &image_file, cv::Mat &mat, bool report_error)
    {
        mat = cv::imread(image_file.string(), cv::IMREAD_COLOR);
        if (mat.empty())
        {
            if (report_error)
            {
                fprintf(stderr, "Read image failed: %s\n", image_file.string().c_str());
            }
            return false;
        }
        return true;
    }

    bool prepare_warmup_data(const std::vector<fs::path> &image_files, std::vector<uint8_t> &data,
                             int input_h, int input_w)
    {
        cv::Mat mat;
        for (const auto &image_file : image_files)
        {
            try
            {
                if (load_image(image_file, mat, false))
                {
                    common::get_input_data_letterbox(mat, data, input_h, input_w);
                    return true;
                }
            }
            catch (const std::exception &)
            {
            }
        }

        return false;
    }
} // namespace

namespace ax
{
    bool post_process(const ax_runner_tensor_t *output, const int nOutputSize, const cv::Mat &mat,
                      int input_w, int input_h, std::vector<detection::Object> &objects)
    {
        if (nOutputSize < 6)
        {
            fprintf(stderr, "Unexpected model output count: %d, expected at least 6.\n", nOutputSize);
            return false;
        }

        std::vector<detection::Object> proposals;

        float* output_ptr[3] = {(float*)output[0].pVirAddr,      // 1*80*80*4
                                (float*)output[2].pVirAddr,      // 1*40*40*4
                                (float*)output[4].pVirAddr};     // 1*20*20*4
        float* output_cls_ptr[3] = {(float*)output[1].pVirAddr,  // 1*80*80*80
                                    (float*)output[3].pVirAddr,  // 1*40*40*80
                                    (float*)output[5].pVirAddr}; // 1*20*20*80
        for (int i = 0; i < 3; ++i)
        {
            auto feat_ptr = output_ptr[i];
            auto feat_cls_ptr = output_cls_ptr[i];
            int32_t stride = (1 << i) * 8;
            detection::generate_proposals_yolo26(stride, feat_ptr, feat_cls_ptr, PROB_THRESHOLD, proposals, input_w, input_h, NUM_CLASS);
        }

        detection::get_out_bbox(proposals, objects, NMS_THRESHOLD, input_h, input_w, mat.rows, mat.cols);
        return true;
    }

    bool recognize_image(ax_runner_axcl &runner, const fs::path &image_file, cv::Mat &mat,
                         std::vector<detection::Object> &objects, std::vector<uint8_t> &data,
                         int input_h, int input_w, RecognitionTiming &timing)
    {
        const auto input_start = Clock::now();
        const auto load_start = Clock::now();
        const bool load_success = load_image(image_file, mat, true);
        timing.image_load_ms = elapsed_ms(load_start, Clock::now());
        if (!load_success)
        {
            timing.input_to_result_ms = elapsed_ms(input_start, Clock::now());
            return false;
        }

        const auto recognition_start = Clock::now();
        const auto finish_recognition_timing = [&]()
        {
            const auto recognition_end = Clock::now();
            timing.recognition_pipeline_ms = elapsed_ms(recognition_start, recognition_end);
            timing.input_to_result_ms = elapsed_ms(input_start, recognition_end);
        };

        const auto preprocess_start = Clock::now();
        common::get_input_data_letterbox(mat, data, input_h, input_w);
        timing.preprocess_ms = elapsed_ms(preprocess_start, Clock::now());

        const auto input_copy_start = Clock::now();
        memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());
        timing.input_copy_ms = elapsed_ms(input_copy_start, Clock::now());

        const auto inference_start = Clock::now();
        const int ret = runner.inference();
        timing.inference_call_ms = elapsed_ms(inference_start, Clock::now());
        if (ret != 0)
        {
            finish_recognition_timing();
            fprintf(stderr, "Inference failed for %s, ret=0x%x.\n", image_file.string().c_str(), ret);
            return false;
        }
        timing.host_to_device_ms = runner.cost_host_to_device;
        timing.model_inference_ms = runner.get_inference_time();
        timing.device_to_host_ms = runner.cost_device_to_host;

        const auto postprocess_start = Clock::now();
        const bool postprocess_success = post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), mat,
                                                      input_w, input_h, objects);
        timing.postprocess_ms = elapsed_ms(postprocess_start, Clock::now());
        finish_recognition_timing();
        if (!postprocess_success)
        {
            return false;
        }

        return true;
    }

    bool run_model(const std::string &model, const std::vector<fs::path> &image_files, const fs::path &output_dir,
                   const std::vector<uint8_t> &warmup_data, int input_h, int input_w)
    {
        ax_runner_axcl runner;
        int ret = runner.init(model.c_str());
        if (ret != 0)
        {
            fprintf(stderr, "init ax model runner failed.\n");
            return false;
        }

        if (runner.get_num_inputs() < 1)
        {
            fprintf(stderr, "Model does not have an input tensor.\n");
            runner.release();
            return false;
        }

        const int model_input_size = runner.get_input(0).nSize;
        if (model_input_size <= 0 || static_cast<size_t>(model_input_size) != warmup_data.size())
        {
            fprintf(stderr, "Model input size mismatch: model=%d, image=%zu.\n", model_input_size, warmup_data.size());
            runner.release();
            return false;
        }

        memcpy(runner.get_input(0).pVirAddr, warmup_data.data(), warmup_data.size());
        fprintf(stdout, "Warm up %d times...\n", WARMUP_COUNT);
        for (int i = 0; i < WARMUP_COUNT; ++i)
        {
            ret = runner.inference();
            if (ret != 0)
            {
                fprintf(stderr, "Warm-up inference failed, ret=0x%x.\n", ret);
                runner.release();
                return false;
            }
        }

        std::vector<uint8_t> data(warmup_data.size(), 0);
        std::vector<RecognitionTiming> recognition_time_costs;
        std::vector<detection::DrawObjectsTiming> output_time_costs;
        size_t recognition_failed_count = 0;
        size_t output_failed_count = 0;

        for (size_t i = 0; i < image_files.size(); ++i)
        {
            const auto &image_file = image_files[i];
            const fs::path output_file = output_dir / image_file.filename();
            fprintf(stdout, "\n[%zu/%zu] 图片：%s\n", i + 1, image_files.size(), image_file.string().c_str());

            RecognitionTiming timing;
            cv::Mat mat;
            std::vector<detection::Object> objects;
            bool recognition_success = false;
            try
            {
                recognition_success = recognize_image(runner, image_file, mat, objects, data,
                                                      input_h, input_w, timing);
            }
            catch (const std::exception &exception)
            {
                fprintf(stderr, "Recognize image failed: %s (%s)\n", image_file.string().c_str(), exception.what());
            }

            print_recognition_timing(timing, recognition_success);
            if (!recognition_success)
            {
                ++recognition_failed_count;
                fprintf(stdout, "输出处理：未执行（识别失败）\n");
                continue;
            }

            recognition_time_costs.push_back(timing);
            fprintf(stdout, "检测目标数：%zu\n", objects.size());

            detection::DrawObjectsTiming output_timing;
            bool output_success = false;
            try
            {
                const auto output_path = output_file.string();
                output_success = detection::draw_objects(mat, objects, CLASS_NAMES, output_path.c_str(),
                                                         0.5, 1, false, &output_timing);
                if (!output_success)
                {
                    fprintf(stderr, "Write result failed: %s\n", output_file.string().c_str());
                }
            }
            catch (const std::exception &exception)
            {
                fprintf(stderr, "Write result failed: %s (%s)\n", output_file.string().c_str(), exception.what());
            }

            print_output_timing(output_timing, output_success);
            if (output_success)
            {
                output_time_costs.push_back(output_timing);
                fprintf(stdout, "结果文件：%s\n", output_file.string().c_str());
            }
            else
            {
                ++output_failed_count;
            }
        }

        fprintf(stdout, "\n--------------------------------------\n");
        fprintf(stdout, "批处理汇总：识别成功 %zu，识别失败 %zu，输出成功 %zu，输出失败 %zu\n",
                recognition_time_costs.size(), recognition_failed_count,
                output_time_costs.size(), output_failed_count);
        if (!recognition_time_costs.empty())
        {
            fprintf(stdout, "识别耗时统计（仅统计识别成功的图片；AXCL 子项已包含在 AXCL 调用总耗时中）：\n");
            print_timing_statistics("图片读取/解码", recognition_time_costs, &RecognitionTiming::image_load_ms);
            print_timing_statistics("图像预处理", recognition_time_costs, &RecognitionTiming::preprocess_ms);
            print_timing_statistics("输入缓冲区复制", recognition_time_costs, &RecognitionTiming::input_copy_ms);
            print_timing_statistics("AXCL 调用总耗时", recognition_time_costs, &RecognitionTiming::inference_call_ms);
            print_timing_statistics("  主机到设备（H2D）", recognition_time_costs, &RecognitionTiming::host_to_device_ms);
            print_timing_statistics("  模型推理", recognition_time_costs, &RecognitionTiming::model_inference_ms);
            print_timing_statistics("  设备到主机（D2H）", recognition_time_costs, &RecognitionTiming::device_to_host_ms);
            print_timing_statistics("后处理", recognition_time_costs, &RecognitionTiming::postprocess_ms);
            print_timing_statistics("识别流程耗时（不含图片读取和输出处理）", recognition_time_costs,
                                    &RecognitionTiming::recognition_pipeline_ms);
            print_timing_statistics("输入到结果耗时（不含绘制和保存）", recognition_time_costs,
                                    &RecognitionTiming::input_to_result_ms);
        }
        if (!output_time_costs.empty())
        {
            fprintf(stdout, "输出处理耗时统计（仅统计输出成功的图片，不计入识别耗时）：\n");
            print_timing_statistics("结果绘制", output_time_costs, &detection::DrawObjectsTiming::render_ms);
            print_timing_statistics("结果保存", output_time_costs, &detection::DrawObjectsTiming::save_ms);
        }
        fprintf(stdout, "--------------------------------------\n");

        runner.release();
        return recognition_failed_count == 0 && output_failed_count == 0 && !recognition_time_costs.empty();
    }
} // namespace ax

int main(int argc, char *argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", false, DEFAULT_MODEL_FILE);
    cmd.add<std::string>("input-dir", 'i', "input image directory", false, DEFAULT_INPUT_DIR);
    cmd.add<std::string>("size", 'g', "input_h, input_w", false, std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.parse_check(argc, argv);

    auto model_file = cmd.get<std::string>("model");
    fs::path input_dir = cmd.get<std::string>("input-dir");
    std::error_code filesystem_error;
    input_dir = fs::absolute(input_dir, filesystem_error).lexically_normal();
    if (filesystem_error)
    {
        fprintf(stderr, "Resolve input directory failed: %s\n", filesystem_error.message().c_str());
        return -1;
    }
    if (input_dir.filename().empty() && input_dir != input_dir.root_path())
    {
        input_dir = input_dir.parent_path();
    }

    if (!utilities::file_exist(model_file))
    {
        fprintf(stderr, "Input model file does not exist: %s\n", model_file.c_str());
        return -1;
    }

    filesystem_error.clear();
    if (!fs::is_directory(input_dir, filesystem_error))
    {
        fprintf(stderr, "Input directory does not exist or is not a directory: %s\n", input_dir.string().c_str());
        return -1;
    }

    const fs::path output_dir = input_dir.parent_path() / DEFAULT_OUTPUT_DIR_NAME;
    filesystem_error.clear();
    fs::create_directories(output_dir, filesystem_error);
    if (filesystem_error)
    {
        fprintf(stderr, "Create output directory failed: %s (%s)\n", output_dir.string().c_str(), filesystem_error.message().c_str());
        return -1;
    }

    filesystem_error.clear();
    if (fs::equivalent(input_dir, output_dir, filesystem_error) && !filesystem_error)
    {
        fprintf(stderr, "Input and output directories must be different: %s\n", input_dir.string().c_str());
        return -1;
    }

    std::vector<fs::path> image_files;
    if (!collect_image_files(input_dir, image_files))
    {
        return -1;
    }
    if (image_files.empty())
    {
        fprintf(stderr, "No supported images found in: %s\n", input_dir.string().c_str());
        return -1;
    }

    const auto input_size_string = cmd.get<std::string>("size");
    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};
    bool input_size_valid = false;
    try
    {
        input_size_valid = utilities::parse_string(input_size_string, input_size);
    }
    catch (const std::exception &)
    {
    }

    if (!input_size_valid || input_size[0] <= 0 || input_size[1] <= 0)
    {
        fprintf(stderr, "Input size is not allowed: %s\n", input_size_string.c_str());
        return -1;
    }

    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "input dir : %s\n", input_dir.string().c_str());
    fprintf(stdout, "output dir : %s\n", output_dir.string().c_str());
    fprintf(stdout, "image count : %zu\n", image_files.size());
    fprintf(stdout, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "--------------------------------------\n");

    const size_t input_data_size = static_cast<size_t>(input_size[0]) * static_cast<size_t>(input_size[1]) * 3;
    std::vector<uint8_t> warmup_data(input_data_size, 0);
    if (!prepare_warmup_data(image_files, warmup_data, input_size[0], input_size[1]))
    {
        fprintf(stderr, "No image can be decoded for warm-up in: %s\n", input_dir.string().c_str());
        return -1;
    }

    if (const auto ret = axclInit(0); 0 != ret)
    {
        fprintf(stderr, "Init AXCL failed{0x%8x}.\n", ret);
        return -1;
    }

    axclrtDeviceList device_list{};
    if (const auto ret = axclrtGetDeviceList(&device_list); 0 != ret || 0 == device_list.num)
    {
        fprintf(stderr, "Get AXCL device failed{0x%8x}, find total %d device.\n", ret, device_list.num);
        axclFinalize();
        return -1;
    }
    if (const auto ret = axclrtSetDevice(device_list.devices[0]); 0 != ret)
    {
        fprintf(stderr, "Set AXCL device failed{0x%8x}.\n", ret);
        axclFinalize();
        return -1;
    }
    if (const auto ret = axclrtEngineInit(AXCL_VNPU_DISABLE); 0 != ret)
    {
        fprintf(stderr, "axclrtEngineInit %d\n", ret);
        axclFinalize();
        return ret;
    }

    const bool success = ax::run_model(model_file, image_files, output_dir, warmup_data, input_size[0], input_size[1]);
    axclFinalize();
    return success ? 0 : -1;
}

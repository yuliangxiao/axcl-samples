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
#include <numeric>
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

const char *DEFAULT_MODEL_FILE = R"(D:\yolo26\yolo26x.axmodel)";
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

    bool load_and_preprocess(const fs::path &image_file, cv::Mat &mat, std::vector<uint8_t> &data,
                             int input_h, int input_w, bool report_error)
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

        common::get_input_data_letterbox(mat, data, input_h, input_w);
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
                if (load_and_preprocess(image_file, mat, data, input_h, input_w, false))
                {
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
                      int input_w, int input_h, const fs::path &output_file)
    {
        if (nOutputSize < 6)
        {
            fprintf(stderr, "Unexpected model output count: %d, expected at least 6.\n", nOutputSize);
            return false;
        }

        std::vector<detection::Object> proposals;
        std::vector<detection::Object> objects;

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
        fprintf(stdout, "detection num: %zu\n", objects.size());

        const auto output_path = output_file.string();
        return detection::draw_objects(mat, objects, CLASS_NAMES, output_path.c_str(), 0.5, 1, false);
    }

    bool recognize_image(ax_runner_axcl &runner, const fs::path &image_file, const fs::path &output_file,
                         std::vector<uint8_t> &data, int input_h, int input_w)
    {
        cv::Mat mat;
        if (!load_and_preprocess(image_file, mat, data, input_h, input_w, true))
        {
            return false;
        }

        memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());
        const int ret = runner.inference();
        if (ret != 0)
        {
            fprintf(stderr, "Inference failed for %s, ret=0x%x.\n", image_file.string().c_str(), ret);
            return false;
        }

        if (!post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), mat, input_w, input_h, output_file))
        {
            fprintf(stderr, "Write result failed: %s\n", output_file.string().c_str());
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
        std::vector<double> time_costs;
        size_t failed_count = 0;

        for (size_t i = 0; i < image_files.size(); ++i)
        {
            const auto &image_file = image_files[i];
            const fs::path output_file = output_dir / image_file.filename();
            fprintf(stdout, "\n[%zu/%zu] image: %s\n", i + 1, image_files.size(), image_file.string().c_str());

            const auto start = std::chrono::steady_clock::now();
            bool success = false;
            try
            {
                success = recognize_image(runner, image_file, output_file, data, input_h, input_w);
            }
            catch (const std::exception &exception)
            {
                fprintf(stderr, "Recognize image failed: %s (%s)\n", image_file.string().c_str(), exception.what());
            }
            const auto end = std::chrono::steady_clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

            fprintf(stdout, "full recognition time: %.3f ms (%s)\n", elapsed_ms, success ? "success" : "failed");
            if (success)
            {
                fprintf(stdout, "result file: %s\n", output_file.string().c_str());
                time_costs.push_back(elapsed_ms);
            }
            else
            {
                ++failed_count;
            }
        }

        fprintf(stdout, "\n--------------------------------------\n");
        fprintf(stdout, "batch summary: success %zu, failed %zu\n", time_costs.size(), failed_count);
        if (!time_costs.empty())
        {
            const double total_time = std::accumulate(time_costs.begin(), time_costs.end(), 0.0);
            const auto min_max_time = std::minmax_element(time_costs.begin(), time_costs.end());
            fprintf(stdout, "full recognition time: avg %.3f ms, min %.3f ms, max %.3f ms\n",
                    total_time / static_cast<double>(time_costs.size()), *min_max_time.first, *min_max_time.second);
        }
        fprintf(stdout, "--------------------------------------\n");

        runner.release();
        return failed_count == 0 && !time_costs.empty();
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

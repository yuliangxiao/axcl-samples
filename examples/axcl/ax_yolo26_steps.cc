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
 * Note: For the YOLO26 series exported by the ultralytics project.
 * Author: LittleMouse
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <opencv2/opencv.hpp>

#if CV_VERSION_MAJOR > 4 || \
    (CV_VERSION_MAJOR == 4 && (CV_VERSION_MINOR > 5 || (CV_VERSION_MINOR == 5 && CV_VERSION_REVISION >= 2)))
#define AXCL_OPENCV_CAPTURE_TIMEOUT_SUPPORTED 1
#else
#define AXCL_OPENCV_CAPTURE_TIMEOUT_SUPPORTED 0
#endif

#include "base/common.hpp"
#include "base/detection.hpp"
#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "yolo26_defaults.hpp"

#include <axcl.h>
#include "ax_model_runner/ax_model_runner_axcl.hpp"

constexpr int DEFAULT_IMG_H = 640;
constexpr int DEFAULT_IMG_W = 640;
constexpr int WARMUP_COUNT = 5;
constexpr int PREVIEW_MAX_WIDTH = 1280;
constexpr int PREVIEW_MAX_HEIGHT = 720;
constexpr int CAPTURE_TIMEOUT_MS = 5000;

const char *WINDOW_NAME = "AXCL YOLO26 RTSP";

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

constexpr int NUM_CLASS = 80;
constexpr float PROB_THRESHOLD = 0.45f;
constexpr float NMS_THRESHOLD = 0.45f;

namespace fs = std::filesystem;

namespace
{
    using Clock = std::chrono::steady_clock;

    struct RunningMetric
    {
        uint64_t count = 0;
        double total = 0.0;
        double minimum = std::numeric_limits<double>::max();
        double maximum = 0.0;

        void add(double value)
        {
            if (!std::isfinite(value) || value < 0.0)
            {
                return;
            }

            ++count;
            total += value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }

        double average() const
        {
            return count == 0 ? 0.0 : total / static_cast<double>(count);
        }
    };

    struct RecognitionTiming
    {
        double preprocess_ms = 0.0;
        double input_copy_ms = 0.0;
        double inference_call_ms = 0.0;
        double host_to_device_ms = 0.0;
        double model_inference_ms = 0.0;
        double device_to_host_ms = 0.0;
        double postprocess_ms = 0.0;
        double recognition_pipeline_ms = 0.0;
    };

    struct FrameTiming
    {
        double latest_frame_copy_ms = 0.0;
        RecognitionTiming recognition;
        double render_display_ms = 0.0;
        double consumer_pipeline_ms = 0.0;
    };

    struct PerformanceStatistics
    {
        uint64_t processed_frames = 0;
        uint64_t dropped_frames = 0;
        RunningMetric latest_frame_copy_ms;
        RunningMetric preprocess_ms;
        RunningMetric input_copy_ms;
        RunningMetric inference_call_ms;
        RunningMetric host_to_device_ms;
        RunningMetric model_inference_ms;
        RunningMetric device_to_host_ms;
        RunningMetric postprocess_ms;
        RunningMetric recognition_pipeline_ms;
        RunningMetric render_display_ms;
        RunningMetric consumer_pipeline_ms;

        void add(const FrameTiming &timing, uint64_t dropped)
        {
            ++processed_frames;
            dropped_frames += dropped;
            latest_frame_copy_ms.add(timing.latest_frame_copy_ms);
            preprocess_ms.add(timing.recognition.preprocess_ms);
            input_copy_ms.add(timing.recognition.input_copy_ms);
            inference_call_ms.add(timing.recognition.inference_call_ms);
            host_to_device_ms.add(timing.recognition.host_to_device_ms);
            model_inference_ms.add(timing.recognition.model_inference_ms);
            device_to_host_ms.add(timing.recognition.device_to_host_ms);
            postprocess_ms.add(timing.recognition.postprocess_ms);
            recognition_pipeline_ms.add(timing.recognition.recognition_pipeline_ms);
            render_display_ms.add(timing.render_display_ms);
            consumer_pipeline_ms.add(timing.consumer_pipeline_ms);
        }
    };

    struct CaptureStatistics
    {
        uint64_t decoded_frames = 0;
        RunningMetric read_ms;
    };

    struct CaptureInformation
    {
        std::string backend = "unknown";
        bool used_automatic_fallback = false;
        bool timeout_supported = AXCL_OPENCV_CAPTURE_TIMEOUT_SUPPORTED != 0;
        double reported_fps = 0.0;
    };

    enum class FrameWaitStatus
    {
        frame_ready,
        timeout,
        stream_ended
    };

    double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::string path_for_log(const fs::path &path)
    {
        return path.u8string();
    }

    fs::path make_console_log_path()
    {
        const auto now = std::chrono::system_clock::now();
        const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_time{};
#ifdef _WIN32
        const bool has_local_time = localtime_s(&local_time, &now_time) == 0;
#else
        const bool has_local_time = localtime_r(&now_time, &local_time) != nullptr;
#endif

        std::string timestamp = "latest";
        char timestamp_buffer[32]{};
        if (has_local_time &&
            std::strftime(timestamp_buffer, sizeof(timestamp_buffer), "%Y%m%d_%H%M%S", &local_time) > 0)
        {
            timestamp = timestamp_buffer;
        }

        return fs::path("log") / ("axcl_yolo26_console_" + timestamp + ".log");
    }

    bool redirect_console_output()
    {
        std::error_code error;
        fs::create_directories("log", error);
        if (error)
        {
            fprintf(stderr, "Create log directory failed: %s\n", error.message().c_str());
            return false;
        }

        const fs::path log_file = make_console_log_path();
        const std::string log_path = path_for_log(log_file);
        fprintf(stdout, "日志文件：%s\n", log_path.c_str());
        fflush(stdout);
        fflush(stderr);

#ifdef _WIN32
        HANDLE log_handle = CreateFileW(log_file.c_str(), FILE_APPEND_DATA,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                                        FILE_ATTRIBUTE_NORMAL, nullptr);
        if (log_handle == INVALID_HANDLE_VALUE)
        {
            fprintf(stderr, "Open console log failed, Windows error: %lu\n", GetLastError());
            return false;
        }

        const int log_descriptor =
            _open_osfhandle(reinterpret_cast<intptr_t>(log_handle), _O_WRONLY | _O_TEXT);
        if (log_descriptor == -1)
        {
            CloseHandle(log_handle);
            fprintf(stderr, "Open console log descriptor failed.\n");
            return false;
        }

        if (_dup2(log_descriptor, _fileno(stdout)) != 0 ||
            _dup2(log_descriptor, _fileno(stderr)) != 0)
        {
            _close(log_descriptor);
            fprintf(stderr, "Redirect console output failed.\n");
            return false;
        }
        _close(log_descriptor);

        const intptr_t stdout_handle = _get_osfhandle(_fileno(stdout));
        const intptr_t stderr_handle = _get_osfhandle(_fileno(stderr));
        if (stdout_handle == -1 || stderr_handle == -1 ||
            !SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(stdout_handle)) ||
            !SetStdHandle(STD_ERROR_HANDLE, reinterpret_cast<HANDLE>(stderr_handle)))
        {
            fprintf(stderr, "Redirect Windows standard handles failed, error: %lu\n", GetLastError());
            return false;
        }
#else
        if (freopen(log_path.c_str(), "a", stdout) == nullptr)
        {
            fprintf(stderr, "Redirect standard output failed.\n");
            return false;
        }
        if (freopen(log_path.c_str(), "a", stderr) == nullptr)
        {
            fprintf(stdout, "Redirect standard error failed.\n");
            return false;
        }
#endif

        fprintf(stdout, "标准输出和标准错误已重定向到：%s\n", log_path.c_str());
        fflush(stdout);
        return true;
    }

    std::string source_for_log(const std::string &source)
    {
        const auto scheme_end = source.find("://");
        if (scheme_end == std::string::npos)
        {
            return source;
        }

        const size_t credentials_start = scheme_end + 3;
        const auto at = source.find('@', credentials_start);
        if (at == std::string::npos)
        {
            return source;
        }

        const auto password_separator = source.find(':', credentials_start);
        if (password_separator != std::string::npos && password_separator < at)
        {
            return source.substr(0, password_separator + 1) + "***" + source.substr(at);
        }

        return source.substr(0, credentials_start) + "***" + source.substr(at);
    }

    class LatestFrameCapture
    {
    public:
        LatestFrameCapture() = default;
        LatestFrameCapture(const LatestFrameCapture &) = delete;
        LatestFrameCapture &operator=(const LatestFrameCapture &) = delete;

        ~LatestFrameCapture()
        {
            stop();
        }

        bool start(const std::string &source)
        {
            try
            {
                worker_ = std::thread(&LatestFrameCapture::capture_loop, this, source);
            }
            catch (const std::exception &exception)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                error_message_ = std::string("Start capture thread failed: ") + exception.what();
                return false;
            }

            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this]()
            {
                return open_complete_;
            });
            const bool success = open_success_;
            lock.unlock();

            if (!success && worker_.joinable())
            {
                worker_.join();
            }
            return success;
        }

        bool wait_for_first_frame(cv::Mat &frame)
        {
            uint64_t sequence = 0;
            double copy_ms = 0.0;
            while (true)
            {
                const auto status = wait_for_new_frame(0, frame, sequence, copy_ms, std::chrono::milliseconds(100));
                if (status == FrameWaitStatus::frame_ready)
                {
                    return true;
                }
                if (status == FrameWaitStatus::stream_ended)
                {
                    return false;
                }
            }
        }

        FrameWaitStatus wait_for_new_frame(uint64_t last_sequence, cv::Mat &frame, uint64_t &sequence,
                                           double &copy_ms, std::chrono::milliseconds timeout)
        {
            cv::Mat shared_frame;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                const bool ready = condition_.wait_for(lock, timeout, [this, last_sequence]()
                {
                    return sequence_ > last_sequence || stream_failed_ || worker_done_ || stop_requested_.load();
                });
                if (!ready)
                {
                    return FrameWaitStatus::timeout;
                }
                if (stream_failed_ || stop_requested_.load())
                {
                    return FrameWaitStatus::stream_ended;
                }
                if (sequence_ <= last_sequence)
                {
                    return FrameWaitStatus::stream_ended;
                }

                shared_frame = latest_frame_;
                sequence = sequence_;
            }

            const auto copy_start = Clock::now();
            shared_frame.copyTo(frame);
            copy_ms = elapsed_ms(copy_start, Clock::now());
            return frame.empty() ? FrameWaitStatus::stream_ended : FrameWaitStatus::frame_ready;
        }

        uint64_t begin_statistics()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            interval_statistics_ = CaptureStatistics{};
            total_statistics_ = CaptureStatistics{};
            return sequence_;
        }

        CaptureStatistics take_interval_statistics()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const CaptureStatistics result = interval_statistics_;
            interval_statistics_ = CaptureStatistics{};
            return result;
        }

        CaptureStatistics total_statistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return total_statistics_;
        }

        CaptureInformation information() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return information_;
        }

        std::string error_message() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return error_message_;
        }

        void stop()
        {
            stop_requested_.store(true);
            condition_.notify_all();
            if (worker_.joinable())
            {
                worker_.join();
            }
        }

    private:
        void report_failure(const std::string &message)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                error_message_ = message;
                stream_failed_ = true;
                worker_done_ = true;
                if (!open_complete_)
                {
                    open_complete_ = true;
                    open_success_ = false;
                }
            }
            condition_.notify_all();
        }

        void capture_loop(const std::string &source)
        {
            cv::VideoCapture capture;
            bool opened_with_ffmpeg = false;
            try
            {
#if AXCL_OPENCV_CAPTURE_TIMEOUT_SUPPORTED
                const std::vector<int> timeout_parameters = {
                    cv::CAP_PROP_OPEN_TIMEOUT_MSEC, CAPTURE_TIMEOUT_MS,
                    cv::CAP_PROP_READ_TIMEOUT_MSEC, CAPTURE_TIMEOUT_MS};
#endif
                const auto try_open = [&](int backend)
                {
                    try
                    {
#if AXCL_OPENCV_CAPTURE_TIMEOUT_SUPPORTED
                        return capture.open(source, backend, timeout_parameters);
#else
                        return capture.open(source, backend);
#endif
                    }
                    catch (const cv::Exception &)
                    {
                        capture.release();
                        return false;
                    }
                };

                opened_with_ffmpeg = try_open(cv::CAP_FFMPEG);
                if (!opened_with_ffmpeg)
                {
                    capture.release();
                    if (!try_open(cv::CAP_ANY))
                    {
                        report_failure("Open RTSP source failed with FFmpeg and automatic OpenCV backends.");
                        return;
                    }
                }

                capture.set(cv::CAP_PROP_BUFFERSIZE, 1.0);

                CaptureInformation information;
                information.used_automatic_fallback = !opened_with_ffmpeg;
                information.reported_fps = capture.get(cv::CAP_PROP_FPS);
                try
                {
                    information.backend = capture.getBackendName();
                }
                catch (const cv::Exception &)
                {
                    information.backend = opened_with_ffmpeg ? "FFMPEG" : "unknown";
                }

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    information_ = std::move(information);
                    open_success_ = true;
                    open_complete_ = true;
                }
                condition_.notify_all();

                while (!stop_requested_.load())
                {
                    cv::Mat frame;
                    const auto read_start = Clock::now();
                    const bool read_success = capture.read(frame);
                    const double read_ms = elapsed_ms(read_start, Clock::now());

                    if (stop_requested_.load())
                    {
                        break;
                    }
                    if (!read_success || frame.empty())
                    {
                        report_failure("RTSP frame read/decode failed.");
                        capture.release();
                        return;
                    }

                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        latest_frame_ = std::move(frame);
                        ++sequence_;
                        ++interval_statistics_.decoded_frames;
                        interval_statistics_.read_ms.add(read_ms);
                        ++total_statistics_.decoded_frames;
                        total_statistics_.read_ms.add(read_ms);
                    }
                    condition_.notify_all();
                }
            }
            catch (const cv::Exception &exception)
            {
                report_failure(std::string("OpenCV capture failed: ") + exception.what());
                return;
            }
            catch (const std::exception &exception)
            {
                report_failure(std::string("RTSP capture failed: ") + exception.what());
                return;
            }

            capture.release();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                worker_done_ = true;
            }
            condition_.notify_all();
        }

        mutable std::mutex mutex_;
        std::condition_variable condition_;
        std::thread worker_;
        std::atomic<bool> stop_requested_{false};
        cv::Mat latest_frame_;
        uint64_t sequence_ = 0;
        bool open_complete_ = false;
        bool open_success_ = false;
        bool stream_failed_ = false;
        bool worker_done_ = false;
        std::string error_message_;
        CaptureInformation information_;
        CaptureStatistics interval_statistics_;
        CaptureStatistics total_statistics_;
    };

    cv::Size preview_size_for(const cv::Mat &frame)
    {
        if (frame.empty())
        {
            return {PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT};
        }

        const double width_scale = static_cast<double>(PREVIEW_MAX_WIDTH) / static_cast<double>(frame.cols);
        const double height_scale = static_cast<double>(PREVIEW_MAX_HEIGHT) / static_cast<double>(frame.rows);
        const double scale = std::min({1.0, width_scale, height_scale});
        return {
            std::max(1, static_cast<int>(std::lround(frame.cols * scale))),
            std::max(1, static_cast<int>(std::lround(frame.rows * scale)))};
    }

    cv::Scalar color_for_label(int label)
    {
        static const std::array<cv::Scalar, 8> COLORS = {
            cv::Scalar(255, 178, 50), cv::Scalar(50, 205, 50), cv::Scalar(255, 90, 90), cv::Scalar(255, 200, 80),
            cv::Scalar(180, 105, 255), cv::Scalar(80, 220, 220), cv::Scalar(220, 160, 80), cv::Scalar(130, 130, 255)};
        const size_t index = label < 0 ? 0 : static_cast<size_t>(label) % COLORS.size();
        return COLORS[index];
    }

    cv::Mat render_result(const cv::Mat &frame, const std::vector<detection::Object> &objects,
                          double actual_fps, double maximum_fps)
    {
        const cv::Size preview_size = preview_size_for(frame);
        cv::Mat preview;
        if (preview_size.width != frame.cols || preview_size.height != frame.rows)
        {
            cv::resize(frame, preview, preview_size, 0.0, 0.0, cv::INTER_LINEAR);
        }
        else
        {
            preview = frame.clone();
        }

        const double scale_x = static_cast<double>(preview.cols) / static_cast<double>(frame.cols);
        const double scale_y = static_cast<double>(preview.rows) / static_cast<double>(frame.rows);
        const double font_scale = std::max(0.45, 0.65 * static_cast<double>(preview.cols) / PREVIEW_MAX_WIDTH);
        constexpr int thickness = 2;

        for (const auto &object : objects)
        {
            const int left = std::clamp(static_cast<int>(std::lround(object.rect.x * scale_x)), 0, preview.cols - 1);
            const int top = std::clamp(static_cast<int>(std::lround(object.rect.y * scale_y)), 0, preview.rows - 1);
            const int right = std::clamp(static_cast<int>(std::lround((object.rect.x + object.rect.width) * scale_x)),
                                         0, preview.cols - 1);
            const int bottom = std::clamp(static_cast<int>(std::lround((object.rect.y + object.rect.height) * scale_y)),
                                          0, preview.rows - 1);
            if (right <= left || bottom <= top)
            {
                continue;
            }

            const cv::Scalar color = color_for_label(object.label);
            cv::rectangle(preview, cv::Rect(left, top, right - left, bottom - top), color, thickness);

            const char *class_name = object.label >= 0 && object.label < NUM_CLASS ? CLASS_NAMES[object.label] : "unknown";
            char label_text[128];
            std::snprintf(label_text, sizeof(label_text), "%s %.1f%%", class_name, object.prob * 100.0f);
            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
                                                       font_scale, 1, &baseline);
            const int text_x = std::clamp(left, 0, std::max(0, preview.cols - text_size.width));
            const int text_y = std::max(0, top - text_size.height - baseline - 4);
            cv::rectangle(preview,
                          cv::Rect(text_x, text_y,
                                   std::min(text_size.width + 4, preview.cols - text_x),
                                   std::min(text_size.height + baseline + 4, preview.rows - text_y)),
                          color, cv::FILLED);
            cv::putText(preview, label_text, cv::Point(text_x + 2, text_y + text_size.height + 1),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        char performance_text[128];
        std::snprintf(performance_text, sizeof(performance_text), "FPS: %.1f | MAX: %.1f", actual_fps, maximum_fps);
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(performance_text, cv::FONT_HERSHEY_SIMPLEX, 0.75, 2, &baseline);
        const int panel_width = std::min(preview.cols, text_size.width + 20);
        const int panel_height = std::min(preview.rows, text_size.height + baseline + 20);
        cv::rectangle(preview, cv::Rect(0, 0, panel_width, panel_height), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(preview, performance_text, cv::Point(10, std::min(preview.rows - 1, text_size.height + 8)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        return preview;
    }

    bool window_is_closed()
    {
        try
        {
            return cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) < 1.0;
        }
        catch (const cv::Exception &)
        {
            return true;
        }
    }

    double frames_per_second(uint64_t frames, double duration_seconds)
    {
        return duration_seconds > 0.0 ? static_cast<double>(frames) / duration_seconds : 0.0;
    }

    double maximum_fps(const RunningMetric &milliseconds)
    {
        const double average = milliseconds.average();
        return average > 0.0 ? 1000.0 / average : 0.0;
    }

    const char *suspected_bottleneck(const PerformanceStatistics &performance, double capture_fps,
                                     double pipeline_max_fps)
    {
        if (performance.processed_frames == 0)
        {
            return "暂无识别样本";
        }
        if (capture_fps <= 0.0)
        {
            return "取流/解码/网络";
        }
        if (pipeline_max_fps < capture_fps * 0.90)
        {
            const double npu_ms = performance.model_inference_ms.average();
            const double non_npu_ms = std::max(0.0, performance.consumer_pipeline_ms.average() - npu_ms);
            return npu_ms >= non_npu_ms ? "NPU 模型执行" : "CPU 处理、传输或显示";
        }
        if (capture_fps < pipeline_max_fps * 0.90)
        {
            return "取流/解码/网络或摄像头帧率上限";
        }
        return "取流与识别吞吐接近";
    }

    void print_interval_statistics(const PerformanceStatistics &performance,
                                   const CaptureStatistics &capture, double duration_seconds,
                                   double &actual_fps, double &pipeline_max_fps)
    {
        actual_fps = frames_per_second(performance.processed_frames, duration_seconds);
        const double capture_fps = frames_per_second(capture.decoded_frames, duration_seconds);
        pipeline_max_fps = maximum_fps(performance.consumer_pipeline_ms);
        const double npu_max_fps = maximum_fps(performance.model_inference_ms);

        fprintf(stdout, "\n[最近 %.2f 秒] 实际 FPS %.2f | MAX %.2f | NPU MAX %.2f | 丢弃 %llu 帧\n",
                duration_seconds, actual_fps, pipeline_max_fps, npu_max_fps,
                static_cast<unsigned long long>(performance.dropped_frames));
        fprintf(stdout, "  取流/网络等待/CPU 解码：%.2f FPS，平均 %.3f ms/帧\n",
                capture_fps, capture.read_ms.average());
        fprintf(stdout, "  最新帧复制：            %8.3f ms\n", performance.latest_frame_copy_ms.average());
        fprintf(stdout, "  图像预处理：            %8.3f ms\n", performance.preprocess_ms.average());
        fprintf(stdout, "  输入缓冲区复制：        %8.3f ms\n", performance.input_copy_ms.average());
        fprintf(stdout, "  AXCL 调用总耗时：       %8.3f ms（包含以下 H2D、NPU、D2H）\n",
                performance.inference_call_ms.average());
        fprintf(stdout, "    主机到设备（H2D）：   %8.3f ms\n", performance.host_to_device_ms.average());
        fprintf(stdout, "    NPU 模型执行：        %8.3f ms\n", performance.model_inference_ms.average());
        fprintf(stdout, "    设备到主机（D2H）：   %8.3f ms\n", performance.device_to_host_ms.average());
        fprintf(stdout, "  后处理：                %8.3f ms\n", performance.postprocess_ms.average());
        fprintf(stdout, "  识别流水线：            %8.3f ms\n", performance.recognition_pipeline_ms.average());
        fprintf(stdout, "  绘制与显示：            %8.3f ms\n", performance.render_display_ms.average());
        fprintf(stdout, "  完整消费端流水线：      %8.3f ms\n", performance.consumer_pipeline_ms.average());
        fprintf(stdout, "  疑似瓶颈：%s（启发式判断）\n",
                suspected_bottleneck(performance, capture_fps, pipeline_max_fps));
        fflush(stdout);
    }

    void print_metric_summary(const char *name, const RunningMetric &metric)
    {
        if (metric.count == 0)
        {
            fprintf(stdout, "  %-24s 无样本\n", name);
            return;
        }
        fprintf(stdout, "  %-24s 平均 %8.3f ms，最小 %8.3f ms，最大 %8.3f ms\n",
                name, metric.average(), metric.minimum, metric.maximum);
    }

    void print_final_statistics(const PerformanceStatistics &performance,
                                const CaptureStatistics &capture, double duration_seconds)
    {
        fprintf(stdout, "\n--------------------------------------\n");
        fprintf(stdout, "实时识别汇总：运行 %.2f 秒，识别 %llu 帧，丢弃 %llu 帧\n",
                duration_seconds,
                static_cast<unsigned long long>(performance.processed_frames),
                static_cast<unsigned long long>(performance.dropped_frames));
        fprintf(stdout, "平均实际 FPS：%.2f，平均 MAX：%.2f，平均 NPU MAX：%.2f\n",
                frames_per_second(performance.processed_frames, duration_seconds),
                maximum_fps(performance.consumer_pipeline_ms),
                maximum_fps(performance.model_inference_ms));
        fprintf(stdout, "取流/网络等待/CPU 解码：%.2f FPS\n",
                frames_per_second(capture.decoded_frames, duration_seconds));
        print_metric_summary("取流/解码调用", capture.read_ms);
        print_metric_summary("最新帧复制", performance.latest_frame_copy_ms);
        print_metric_summary("图像预处理", performance.preprocess_ms);
        print_metric_summary("输入缓冲区复制", performance.input_copy_ms);
        print_metric_summary("AXCL 调用总耗时", performance.inference_call_ms);
        print_metric_summary("主机到设备（H2D）", performance.host_to_device_ms);
        print_metric_summary("NPU 模型执行", performance.model_inference_ms);
        print_metric_summary("设备到主机（D2H）", performance.device_to_host_ms);
        print_metric_summary("后处理", performance.postprocess_ms);
        print_metric_summary("识别流水线", performance.recognition_pipeline_ms);
        print_metric_summary("绘制与显示", performance.render_display_ms);
        print_metric_summary("完整消费端流水线", performance.consumer_pipeline_ms);
        fprintf(stdout, "--------------------------------------\n");
    }

    bool prepare_warmup_data(const cv::Mat &frame, std::vector<uint8_t> &data, int input_h, int input_w)
    {
        if (frame.empty())
        {
            return false;
        }

        try
        {
            common::get_input_data_letterbox(frame, data, input_h, input_w);
            return true;
        }
        catch (const std::exception &exception)
        {
            fprintf(stderr, "Prepare warm-up frame failed: %s\n", exception.what());
            return false;
        }
    }
} // namespace

namespace ax
{
    bool post_process(const ax_runner_tensor_t *output, int output_count, const cv::Mat &frame,
                      int input_w, int input_h, std::vector<detection::Object> &objects)
    {
        if (output_count < 6)
        {
            fprintf(stderr, "Unexpected model output count: %d, expected at least 6.\n", output_count);
            return false;
        }

        std::vector<detection::Object> proposals;
        float *output_ptr[3] = {
            static_cast<float *>(output[0].pVirAddr),
            static_cast<float *>(output[2].pVirAddr),
            static_cast<float *>(output[4].pVirAddr)};
        float *output_class_ptr[3] = {
            static_cast<float *>(output[1].pVirAddr),
            static_cast<float *>(output[3].pVirAddr),
            static_cast<float *>(output[5].pVirAddr)};

        for (int i = 0; i < 3; ++i)
        {
            const int32_t stride = (1 << i) * 8;
            detection::generate_proposals_yolo26(stride, output_ptr[i], output_class_ptr[i],
                                                 PROB_THRESHOLD, proposals, input_w, input_h, NUM_CLASS);
        }

        detection::get_out_bbox(proposals, objects, NMS_THRESHOLD, input_h, input_w, frame.rows, frame.cols);
        return true;
    }

    bool recognize_frame(ax_runner_axcl &runner, const cv::Mat &frame,
                         std::vector<detection::Object> &objects, std::vector<uint8_t> &data,
                         int input_h, int input_w, RecognitionTiming &timing)
    {
        const auto recognition_start = Clock::now();

        const auto preprocess_start = Clock::now();
        common::get_input_data_letterbox(frame, data, input_h, input_w);
        timing.preprocess_ms = elapsed_ms(preprocess_start, Clock::now());

        const auto input_copy_start = Clock::now();
        std::memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());
        timing.input_copy_ms = elapsed_ms(input_copy_start, Clock::now());

        const auto inference_start = Clock::now();
        const int ret = runner.inference();
        timing.inference_call_ms = elapsed_ms(inference_start, Clock::now());
        if (ret != 0)
        {
            timing.recognition_pipeline_ms = elapsed_ms(recognition_start, Clock::now());
            fprintf(stderr, "Inference failed, ret=0x%x.\n", ret);
            return false;
        }
        timing.host_to_device_ms = runner.cost_host_to_device;
        timing.model_inference_ms = runner.get_inference_time();
        timing.device_to_host_ms = runner.cost_device_to_host;

        const auto postprocess_start = Clock::now();
        const bool postprocess_success = post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(),
                                                      frame, input_w, input_h, objects);
        timing.postprocess_ms = elapsed_ms(postprocess_start, Clock::now());
        timing.recognition_pipeline_ms = elapsed_ms(recognition_start, Clock::now());
        return postprocess_success;
    }

    bool run_model(const std::string &model, LatestFrameCapture &capture,
                   const std::vector<uint8_t> &warmup_data, const cv::Mat &first_frame,
                   int input_h, int input_w)
    {
        struct CaptureStopGuard
        {
            LatestFrameCapture &capture;

            ~CaptureStopGuard()
            {
                capture.stop();
            }
        } capture_stop_guard{capture};

        ax_runner_axcl runner;
        int ret = runner.init(model.c_str());
        if (ret != 0)
        {
            fprintf(stderr, "Init AX model runner failed.\n");
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
            fprintf(stderr, "Model input size mismatch: model=%d, frame=%zu.\n",
                    model_input_size, warmup_data.size());
            runner.release();
            return false;
        }

        std::memcpy(runner.get_input(0).pVirAddr, warmup_data.data(), warmup_data.size());
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
        PerformanceStatistics interval_statistics;
        PerformanceStatistics total_statistics;
        uint64_t last_sequence = capture.begin_statistics();
        double display_actual_fps = 0.0;
        double display_maximum_fps = 0.0;
        bool failed = false;
        bool user_requested_exit = false;

        try
        {
            cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            const cv::Size initial_preview_size = preview_size_for(first_frame);
            cv::resizeWindow(WINDOW_NAME, initial_preview_size.width, initial_preview_size.height);
        }
        catch (const cv::Exception &exception)
        {
            fprintf(stderr, "Create OpenCV preview window failed: %s\n", exception.what());
            runner.release();
            return false;
        }

        const auto benchmark_start = Clock::now();
        auto interval_start = benchmark_start;

        while (!failed && !user_requested_exit)
        {
            cv::Mat frame;
            uint64_t sequence = last_sequence;
            double latest_frame_copy_ms = 0.0;
            const auto wait_status = capture.wait_for_new_frame(last_sequence, frame, sequence,
                                                                 latest_frame_copy_ms,
                                                                 std::chrono::milliseconds(10));
            if (wait_status == FrameWaitStatus::stream_ended)
            {
                fprintf(stderr, "RTSP stream ended: %s\n", capture.error_message().c_str());
                failed = true;
                break;
            }

            if (wait_status == FrameWaitStatus::timeout)
            {
                const int key = cv::waitKey(1) & 0xff;
                if (key == 'q' || key == 'Q' || key == 27 || window_is_closed())
                {
                    user_requested_exit = true;
                }
            }
            else
            {
                const uint64_t dropped = sequence > last_sequence + 1 ? sequence - last_sequence - 1 : 0;
                last_sequence = sequence;

                FrameTiming timing;
                timing.latest_frame_copy_ms = latest_frame_copy_ms;
                const auto consumer_start = Clock::now();
                std::vector<detection::Object> objects;

                try
                {
                    if (!recognize_frame(runner, frame, objects, data, input_h, input_w, timing.recognition))
                    {
                        failed = true;
                        break;
                    }

                    const auto render_start = Clock::now();
                    cv::Mat preview = render_result(frame, objects, display_actual_fps, display_maximum_fps);
                    cv::imshow(WINDOW_NAME, preview);
                    const int key = cv::waitKey(1) & 0xff;
                    timing.render_display_ms = elapsed_ms(render_start, Clock::now());
                    timing.consumer_pipeline_ms = timing.latest_frame_copy_ms + elapsed_ms(consumer_start, Clock::now());

                    interval_statistics.add(timing, dropped);
                    total_statistics.add(timing, dropped);

                    if (key == 'q' || key == 'Q' || key == 27 || window_is_closed())
                    {
                        user_requested_exit = true;
                    }
                }
                catch (const cv::Exception &exception)
                {
                    fprintf(stderr, "OpenCV frame processing/display failed: %s\n", exception.what());
                    failed = true;
                }
                catch (const std::exception &exception)
                {
                    fprintf(stderr, "Frame recognition failed: %s\n", exception.what());
                    failed = true;
                }
            }

            const auto now = Clock::now();
            const double interval_seconds = elapsed_ms(interval_start, now) / 1000.0;
            if (interval_seconds >= 1.0)
            {
                const CaptureStatistics capture_interval = capture.take_interval_statistics();
                print_interval_statistics(interval_statistics, capture_interval, interval_seconds,
                                          display_actual_fps, display_maximum_fps);
                interval_statistics = PerformanceStatistics{};
                interval_start = now;
            }
        }

        const auto benchmark_end = Clock::now();
        const CaptureStatistics capture_total = capture.total_statistics();
        capture.stop();
        try
        {
            cv::destroyWindow(WINDOW_NAME);
        }
        catch (const cv::Exception &)
        {
        }

        print_final_statistics(total_statistics, capture_total,
                               elapsed_ms(benchmark_start, benchmark_end) / 1000.0);
        runner.release();
        return !failed && total_statistics.processed_frames > 0;
    }
} // namespace ax

int main(int argc, char *argv[])
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    if (!redirect_console_output())
    {
        return -1;
    }

    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", false, yolo26_defaults::kModelPath);
    cmd.add<std::string>("source", 's', "RTSP source URL", false, yolo26_defaults::kRtspSource);
    cmd.add<std::string>("size", 'g', "input_h, input_w", false,
                         std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.parse_check(argc, argv);

    const std::string model_file = cmd.get<std::string>("model");
    const std::string source = cmd.get<std::string>("source");
    const std::string input_size_string = cmd.get<std::string>("size");

    if (!utilities::file_exist(model_file))
    {
        fprintf(stderr, "Input model file does not exist: %s\n", path_for_log(fs::path(model_file)).c_str());
        return -1;
    }
    if (source.empty())
    {
        fprintf(stderr, "RTSP source must not be empty.\n");
        return -1;
    }

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
    fprintf(stdout, "model file : %s\n", path_for_log(fs::path(model_file)).c_str());
    fprintf(stdout, "RTSP source : %s\n", source_for_log(source).c_str());
    fprintf(stdout, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "preview max : %d x %d\n", PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT);
    fprintf(stdout, "exit keys : Q / Esc\n");
    fprintf(stdout, "--------------------------------------\n");

    LatestFrameCapture capture;
    if (!capture.start(source))
    {
        fprintf(stderr, "Open RTSP source failed: %s\n", capture.error_message().c_str());
        return -1;
    }

    cv::Mat first_frame;
    if (!capture.wait_for_first_frame(first_frame))
    {
        fprintf(stderr, "Read first RTSP frame failed: %s\n", capture.error_message().c_str());
        return -1;
    }

    const CaptureInformation capture_information = capture.information();
    fprintf(stdout, "OpenCV backend : %s%s\n", capture_information.backend.c_str(),
            capture_information.used_automatic_fallback ? " (automatic fallback)" : "");
    if (capture_information.timeout_supported)
    {
        fprintf(stdout, "requested capture timeout : %d ms\n", CAPTURE_TIMEOUT_MS);
    }
    else
    {
        fprintf(stdout, "requested capture timeout : unavailable in this OpenCV version\n");
    }
    fprintf(stdout, "stream frame : %d x %d\n", first_frame.cols, first_frame.rows);
    if (capture_information.reported_fps > 0.0)
    {
        fprintf(stdout, "reported stream FPS : %.2f (backend metadata, may be inaccurate)\n",
                capture_information.reported_fps);
    }

    const size_t input_data_size = static_cast<size_t>(input_size[0]) * static_cast<size_t>(input_size[1]) * 3;
    std::vector<uint8_t> warmup_data(input_data_size, 0);
    if (!prepare_warmup_data(first_frame, warmup_data, input_size[0], input_size[1]))
    {
        return -1;
    }

    if (const auto ret = axclInit(0); ret != 0)
    {
        fprintf(stderr, "Init AXCL failed{0x%8x}.\n", ret);
        return -1;
    }

    axclrtDeviceList device_list{};
    if (const auto ret = axclrtGetDeviceList(&device_list); ret != 0 || device_list.num == 0)
    {
        fprintf(stderr, "Get AXCL device failed{0x%8x}, find total %d device.\n", ret, device_list.num);
        axclFinalize();
        return -1;
    }
    if (const auto ret = axclrtSetDevice(device_list.devices[0]); ret != 0)
    {
        fprintf(stderr, "Set AXCL device failed{0x%8x}.\n", ret);
        axclFinalize();
        return -1;
    }
    if (const auto ret = axclrtEngineInit(AXCL_VNPU_DISABLE); ret != 0)
    {
        fprintf(stderr, "axclrtEngineInit %d\n", ret);
        axclFinalize();
        return ret;
    }

    const bool success = ax::run_model(model_file, capture, warmup_data, first_frame,
                                       input_size[0], input_size[1]);
    axclFinalize();
    return success ? 0 : -1;
}

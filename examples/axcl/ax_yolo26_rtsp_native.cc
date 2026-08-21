/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2026, AXERA Semiconductor Co., Ltd. All rights reserved.
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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <conio.h>
#include <fcntl.h>
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
#include <libavutil/time.h>
}

#include <opencv2/opencv.hpp>

#include <ax_buffer_tool.h>
#include <axcl.h>
#include <axcl_ivps.h>
#include <axcl_rt_context.h>
#include <axcl_rt_device.h>
#include <axcl_rt_memory.h>
#include <axcl_sys.h>
#include <axcl_vdec.h>

#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "base/detection.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "yolo26_defaults.hpp"

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kSourceWidth = 2560;
constexpr int kSourceHeight = 1440;
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;
constexpr int kInputChannels = 3;
constexpr std::size_t kInputStride = static_cast<std::size_t>(kInputWidth) * kInputChannels;
constexpr std::size_t kInputBytes = kInputStride * kInputHeight;
constexpr int kModelGroupId = 0;
constexpr int kWarmupCount = 5;
constexpr int kClassCount = 80;
constexpr float kProbabilityThreshold = 0.45F;
constexpr float kNmsThreshold = 0.45F;
constexpr AX_VDEC_CHN kVdecChannel = 0;
constexpr AX_U32 kH264FrameBufferCount = 32;
constexpr AX_S32 kAxWaitMs = 100;
constexpr double kSlowVdecSendMilliseconds = 50.0;
constexpr AX_S32 kAxclRuntimeTaskTimeout =
    AXCL_DEF_RUNTIME_ERR(AXCL_RUNTIME_TASK, AXCL_ERR_TIMEOUT);

enum class ApplicationLogLevel {
    kInfo,
    kWarning,
    kError,
};

std::mutex g_application_log_mutex;
std::array<char, 64U * 1024U> g_application_log_buffer{};
FILE* g_original_standard_error = nullptr;
std::string g_application_log_path;
bool g_console_log_path_reported = false;

const char* ApplicationLogLevelName(ApplicationLogLevel level) {
    switch (level) {
    case ApplicationLogLevel::kInfo:
        return "INFO";
    case ApplicationLogLevel::kWarning:
        return "WARN";
    case ApplicationLogLevel::kError:
        return "ERROR";
    }
    return "UNKNOWN";
}

std::string WallClockTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  now.time_since_epoch()) %
                              1000;
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    const bool converted = localtime_s(&local_time, &now_time) == 0;
#else
    const bool converted = localtime_r(&now_time, &local_time) != nullptr;
#endif
    if (!converted) {
        return "0000-00-00 00:00:00.000";
    }

    char date_time[32]{};
    if (std::strftime(date_time, sizeof(date_time), "%Y-%m-%d %H:%M:%S", &local_time) == 0) {
        return "0000-00-00 00:00:00.000";
    }

    char timestamp[40]{};
    std::snprintf(timestamp, sizeof(timestamp), "%s.%03lld", date_time,
                  static_cast<long long>(milliseconds.count()));
    return timestamp;
}

std::string FormatLogMessage(const char* format, va_list arguments) {
    va_list size_arguments;
    va_copy(size_arguments, arguments);
    const int required = std::vsnprintf(nullptr, 0, format, size_arguments);
    va_end(size_arguments);
    if (required < 0) {
        return "<log formatting failed>";
    }

    std::vector<char> buffer(static_cast<std::size_t>(required) + 1U);
    va_list write_arguments;
    va_copy(write_arguments, arguments);
    const int written = std::vsnprintf(buffer.data(), buffer.size(), format, write_arguments);
    va_end(write_arguments);
    if (written < 0) {
        return "<log formatting failed>";
    }
    return std::string(buffer.data(), static_cast<std::size_t>(written));
}

void WriteApplicationLog(ApplicationLogLevel level, bool flush, const char* format, va_list arguments) {
    const std::string message = FormatLogMessage(format, arguments);
    const std::string timestamp = WallClockTimestamp();
    const char* level_name = ApplicationLogLevelName(level);

    std::lock_guard<std::mutex> lock(g_application_log_mutex);
    std::fprintf(stdout, "[%s][%s] %s\n", timestamp.c_str(), level_name, message.c_str());
    if (flush || level != ApplicationLogLevel::kInfo) {
        std::fflush(stdout);
    }

    if (level == ApplicationLogLevel::kError && g_original_standard_error != nullptr) {
        std::fprintf(g_original_standard_error, "[%s][%s] %s\n",
                     timestamp.c_str(), level_name, message.c_str());
        if (!g_console_log_path_reported && !g_application_log_path.empty()) {
            std::fprintf(g_original_standard_error, "[LOG] %s\n", g_application_log_path.c_str());
            g_console_log_path_reported = true;
        }
        std::fflush(g_original_standard_error);
    }
}

void LogInfo(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    WriteApplicationLog(ApplicationLogLevel::kInfo, false, format, arguments);
    va_end(arguments);
}

void LogInfoFlush(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    WriteApplicationLog(ApplicationLogLevel::kInfo, true, format, arguments);
    va_end(arguments);
}

void LogWarning(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    WriteApplicationLog(ApplicationLogLevel::kWarning, true, format, arguments);
    va_end(arguments);
}

void LogError(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    WriteApplicationLog(ApplicationLogLevel::kError, true, format, arguments);
    va_end(arguments);
}

std::string PathForLog(const fs::path& path) {
    return path.u8string();
}

std::uint32_t CurrentProcessId() {
#ifdef _WIN32
    return static_cast<std::uint32_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint32_t>(getpid());
#endif
}

fs::path MakeApplicationLogPath(unsigned int collision_index) {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  now.time_since_epoch()) %
                              1000;
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    const bool converted = localtime_s(&local_time, &now_time) == 0;
#else
    const bool converted = localtime_r(&now_time, &local_time) != nullptr;
#endif

    char date_time[32]{};
    const bool formatted = converted &&
                           std::strftime(date_time, sizeof(date_time), "%Y%m%d_%H%M%S", &local_time) > 0;
    const std::string timestamp = formatted ? date_time : "latest";
    char millisecond_text[4]{};
    std::snprintf(millisecond_text, sizeof(millisecond_text), "%03lld",
                  static_cast<long long>(milliseconds.count()));
    std::string file_name = "ax_yolo26_rtsp_native_" + timestamp + "_" +
                            millisecond_text + "_pid" +
                            std::to_string(CurrentProcessId());
    if (collision_index != 0) {
        file_name += "_" + std::to_string(collision_index);
    }
    return fs::path("log") / (file_name + ".log");
}

#ifdef _WIN32
int DuplicateDescriptor(int descriptor) {
    return _dup(descriptor);
}

int DuplicateToDescriptor(int source, int destination) {
    return _dup2(source, destination);
}

int FileDescriptor(FILE* file) {
    return _fileno(file);
}

int CloseDescriptor(int descriptor) {
    return _close(descriptor);
}

FILE* DescriptorToFile(int descriptor) {
    return _fdopen(descriptor, "w");
}
#else
int DuplicateDescriptor(int descriptor) {
    return dup(descriptor);
}

int DuplicateToDescriptor(int source, int destination) {
    return dup2(source, destination) == -1 ? -1 : 0;
}

int FileDescriptor(FILE* file) {
    return fileno(file);
}

int CloseDescriptor(int descriptor) {
    return close(descriptor);
}

FILE* DescriptorToFile(int descriptor) {
    return fdopen(descriptor, "w");
}
#endif

int OpenNewLogDescriptor(const fs::path& path, bool* already_exists, std::string* error_text) {
    if (already_exists != nullptr) {
        *already_exists = false;
    }
#ifdef _WIN32
    const HANDLE handle = CreateFileW(path.c_str(), FILE_APPEND_DATA,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, CREATE_NEW,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        if (already_exists != nullptr) {
            *already_exists = error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS;
        }
        if (error_text != nullptr) {
            *error_text = "Windows error " + std::to_string(error);
        }
        return -1;
    }
    const int descriptor = _open_osfhandle(reinterpret_cast<intptr_t>(handle),
                                            _O_WRONLY | _O_APPEND | _O_TEXT);
    if (descriptor == -1) {
        CloseHandle(handle);
        if (error_text != nullptr) {
            *error_text = "_open_osfhandle failed";
        }
    }
    return descriptor;
#else
    const std::string native_path = PathForLog(path);
    const int descriptor = open(native_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND, 0644);
    if (descriptor == -1) {
        const int error = errno;
        if (already_exists != nullptr) {
            *already_exists = error == EEXIST;
        }
        if (error_text != nullptr) {
            *error_text = std::error_code(error, std::generic_category()).message();
        }
    }
    return descriptor;
#endif
}

bool RedirectStandardStreams(int log_descriptor) {
    std::fflush(stdout);
    std::fflush(stderr);

    const int stdout_descriptor = FileDescriptor(stdout);
    const int stderr_descriptor = FileDescriptor(stderr);
    const int original_stdout = DuplicateDescriptor(stdout_descriptor);
    const int original_stderr = DuplicateDescriptor(stderr_descriptor);
    if (original_stdout == -1 || original_stderr == -1) {
        if (original_stdout != -1) {
            CloseDescriptor(original_stdout);
        }
        if (original_stderr != -1) {
            CloseDescriptor(original_stderr);
        }
        return false;
    }

#ifdef _WIN32
    const HANDLE original_stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);
    const HANDLE original_stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
#endif

    const bool redirected = DuplicateToDescriptor(log_descriptor, stdout_descriptor) == 0 &&
                            DuplicateToDescriptor(log_descriptor, stderr_descriptor) == 0;
    if (!redirected) {
        (void)DuplicateToDescriptor(original_stdout, stdout_descriptor);
        (void)DuplicateToDescriptor(original_stderr, stderr_descriptor);
        CloseDescriptor(original_stdout);
        CloseDescriptor(original_stderr);
        return false;
    }

#ifdef _WIN32
    const intptr_t stdout_handle = _get_osfhandle(stdout_descriptor);
    const intptr_t stderr_handle = _get_osfhandle(stderr_descriptor);
    if (stdout_handle == -1 || stderr_handle == -1 ||
        !SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(stdout_handle)) ||
        !SetStdHandle(STD_ERROR_HANDLE, reinterpret_cast<HANDLE>(stderr_handle))) {
        (void)DuplicateToDescriptor(original_stdout, stdout_descriptor);
        (void)DuplicateToDescriptor(original_stderr, stderr_descriptor);
        (void)SetStdHandle(STD_OUTPUT_HANDLE, original_stdout_handle);
        (void)SetStdHandle(STD_ERROR_HANDLE, original_stderr_handle);
        CloseDescriptor(original_stdout);
        CloseDescriptor(original_stderr);
        return false;
    }
#endif

    FILE* original_error_file = DescriptorToFile(original_stderr);
    if (original_error_file == nullptr) {
        (void)DuplicateToDescriptor(original_stdout, stdout_descriptor);
        (void)DuplicateToDescriptor(original_stderr, stderr_descriptor);
#ifdef _WIN32
        (void)SetStdHandle(STD_OUTPUT_HANDLE, original_stdout_handle);
        (void)SetStdHandle(STD_ERROR_HANDLE, original_stderr_handle);
#endif
        CloseDescriptor(original_stdout);
        CloseDescriptor(original_stderr);
        return false;
    }

    CloseDescriptor(original_stdout);
    g_original_standard_error = original_error_file;
    (void)std::setvbuf(g_original_standard_error, nullptr, _IONBF, 0);
    return true;
}

bool InitializeApplicationLogging() {
    if (std::setvbuf(stdout, g_application_log_buffer.data(), _IOFBF,
                     g_application_log_buffer.size()) != 0) {
        std::fprintf(stderr, "Configure application log buffering failed.\n");
        std::fflush(stderr);
        return false;
    }

    std::error_code directory_error;
    fs::create_directories("log", directory_error);
    if (directory_error) {
        std::fprintf(stderr, "Create log directory failed: %s\n", directory_error.message().c_str());
        std::fflush(stderr);
        return false;
    }

    fs::path log_path;
    int log_descriptor = -1;
    std::string open_error;
    for (unsigned int collision_index = 0; collision_index < 100; ++collision_index) {
        log_path = MakeApplicationLogPath(collision_index);
        bool already_exists = false;
        log_descriptor = OpenNewLogDescriptor(log_path, &already_exists, &open_error);
        if (log_descriptor != -1) {
            break;
        }
        if (!already_exists) {
            break;
        }
    }
    if (log_descriptor == -1) {
        std::fprintf(stderr, "Open application log failed: %s (%s)\n",
                     PathForLog(log_path).c_str(), open_error.c_str());
        std::fflush(stderr);
        return false;
    }

    if (!RedirectStandardStreams(log_descriptor)) {
        CloseDescriptor(log_descriptor);
        std::fprintf(stderr, "Redirect standard output and standard error failed.\n");
        std::fflush(stderr);
        return false;
    }
    CloseDescriptor(log_descriptor);

    std::error_code absolute_error;
    const fs::path absolute_path = fs::absolute(log_path, absolute_error);
    g_application_log_path = PathForLog(absolute_error ? log_path : absolute_path);
    LogInfoFlush("[SYSTEM] application log initialized: %s", g_application_log_path.c_str());
    return true;
}

void FlushApplicationLog() {
    std::lock_guard<std::mutex> lock(g_application_log_mutex);
    std::fflush(stdout);
    std::fflush(stderr);
}

#ifdef _WIN32
bool HasArgument(int argc, char* argv[], const char* expected) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], expected) == 0) {
            return true;
        }
    }
    return false;
}

bool OwnsInteractiveConsole() {
    DWORD process_ids[2]{};
    if (GetConsoleProcessList(process_ids, 2) != 1) {
        return false;
    }
    const HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    return input != nullptr && input != INVALID_HANDLE_VALUE && GetConsoleMode(input, &mode) != 0;
}

void PauseIndependentConsole() {
    FlushApplicationLog();
    FILE* output = g_original_standard_error != nullptr ? g_original_standard_error : stderr;
    std::fprintf(output, "\n程序已结束，按任意键关闭窗口...\n");
    std::fflush(output);
    (void)_getch();
}

void ConfigureExitPause(int argc, char* argv[]) {
    if (!HasArgument(argc, argv, "--no-pause") && OwnsInteractiveConsole()) {
        (void)std::atexit(PauseIndependentConsole);
    }
}
#else
void ConfigureExitPause(int, char*[]) {}
#endif

enum class RunMode {
    kVdecSmoke,
    kIvpsSmoke,
    kInfer,
};

struct Options {
    RunMode mode{RunMode::kInfer};
    std::string source{yolo26_defaults::kRtspSource};
    std::string model{yolo26_defaults::kModelPath};
    std::string axcl_config;
    std::string dump_ivps;
    int device_index{0};
    int duration_seconds{0};
    int read_timeout_ms{5000};
    int statistics_interval_seconds{5};
};

struct InterruptState {
    std::atomic<bool> stop{false};
    std::atomic<std::int64_t> deadline_us{0};
};

InterruptState* g_interrupt_state = nullptr;

#ifdef _WIN32
BOOL WINAPI ConsoleControlHandler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT) {
        if (g_interrupt_state != nullptr) {
            g_interrupt_state->stop.store(true, std::memory_order_relaxed);
        }
        return TRUE;
    }
    return FALSE;
}
#else
void SignalHandler(int) {
    if (g_interrupt_state != nullptr) {
        g_interrupt_state->stop.store(true, std::memory_order_relaxed);
    }
}
#endif

const char* ModeName(RunMode mode) {
    switch (mode) {
    case RunMode::kVdecSmoke:
        return "vdec-smoke";
    case RunMode::kIvpsSmoke:
        return "ivps-smoke";
    case RunMode::kInfer:
        return "infer";
    }
    return "unknown";
}

bool ParseMode(const std::string& value, RunMode* mode) {
    if (mode == nullptr) {
        return false;
    }
    if (value == "vdec-smoke") {
        *mode = RunMode::kVdecSmoke;
        return true;
    }
    if (value == "ivps-smoke") {
        *mode = RunMode::kIvpsSmoke;
        return true;
    }
    if (value == "infer") {
        *mode = RunMode::kInfer;
        return true;
    }
    return false;
}

std::string RedactRtspUrl(const std::string& url) {
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) {
        return "<invalid-rtsp-url>";
    }
    const auto authority_begin = scheme + 3;
    const auto authority_end = url.find('/', authority_begin);
    const auto at = url.find('@', authority_begin);
    if (at == std::string::npos || (authority_end != std::string::npos && at > authority_end)) {
        return url;
    }
    return url.substr(0, authority_begin) + "***:***@" + url.substr(at + 1);
}

std::string AvErrorText(int error) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    if (av_strerror(error, text.data(), text.size()) < 0) {
        return "unknown FFmpeg error";
    }
    return text.data();
}

double ElapsedSeconds(Clock::time_point begin, Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double>(end - begin).count();
}

double ElapsedMilliseconds(Clock::time_point begin, Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

AX_U32 AlignUp(AX_U32 value, AX_U32 alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

int FfmpegInterruptCallback(void* opaque) {
    auto* state = static_cast<InterruptState*>(opaque);
    if (state == nullptr) {
        return 0;
    }
    if (state->stop.load(std::memory_order_relaxed)) {
        return 1;
    }
    const auto deadline = state->deadline_us.load(std::memory_order_relaxed);
    return deadline != 0 && av_gettime_relative() >= deadline ? 1 : 0;
}

void ArmDeadline(InterruptState* state, std::int64_t timeout_us) {
    if (state == nullptr) {
        return;
    }
    const auto deadline = timeout_us > 0 ? av_gettime_relative() + timeout_us : 0;
    state->deadline_us.store(deadline, std::memory_order_relaxed);
}

bool HasAnnexBPrefix(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size < 3) {
        return false;
    }
    std::size_t index = 0;
    while (index < size && data[index] == 0) {
        ++index;
    }
    return index >= 2 && index < size && data[index] == 1;
}

struct H264NalSummary {
    bool has_sps{false};
    bool has_pps{false};
    bool has_idr{false};
    bool has_vcl{false};
};

H264NalSummary InspectAnnexBNals(const std::uint8_t* data, std::size_t size) {
    H264NalSummary summary{};
    if (data == nullptr || size < 4) {
        return summary;
    }

    for (std::size_t i = 0; i + 3 < size; ++i) {
        std::size_t nal = 0;
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
            nal = i + 3;
        } else if (i + 4 < size && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 &&
                   data[i + 3] == 1) {
            nal = i + 4;
        } else {
            continue;
        }
        if (nal >= size) {
            continue;
        }
        const auto type = data[nal] & 0x1FU;
        summary.has_sps = summary.has_sps || type == 7;
        summary.has_pps = summary.has_pps || type == 8;
        summary.has_idr = summary.has_idr || type == 5;
        summary.has_vcl = summary.has_vcl || (type >= 1 && type <= 5);
        i = nal;
    }
    return summary;
}

struct AccessUnit {
    std::vector<std::uint8_t> data;
    std::uint64_t pts_us{0};
};

enum class ReadResult {
    kPacket,
    kEof,
    kInterrupted,
    kError,
};

class FfmpegRtspDemuxer final {
public:
    explicit FfmpegRtspDemuxer(InterruptState* interrupt) : interrupt_(interrupt) {}

    ~FfmpegRtspDemuxer() {
        Close();
    }

    bool Open(const std::string& source, int timeout_ms) {
        Close();
        annex_b_parameter_sets_.clear();
        pre_idr_parameter_sets_.clear();
        bitstream_mode_.clear();
        input_packets_ = 0;
        skipped_before_idr_ = 0;
        ffmpeg_errors_ = 0;
        synthetic_pts_us_ = 0;
        reported_fps_ = 0.0;
        waiting_for_idr_ = true;
        bsf_eof_sent_ = false;
        fatal_packet_error_ = false;
        timeout_us_ = static_cast<std::int64_t>(timeout_ms) * 1000;
        format_ = avformat_alloc_context();
        if (format_ == nullptr) {
            LogError("[FFMPEG] avformat_alloc_context failed");
            return false;
        }
        format_->interrupt_callback.callback = FfmpegInterruptCallback;
        format_->interrupt_callback.opaque = interrupt_;

        AVDictionary* options = nullptr;
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
#if LIBAVFORMAT_VERSION_MAJOR >= 59
        av_dict_set_int(&options, "timeout", timeout_us_, 0);
#else
        av_dict_set_int(&options, "stimeout", timeout_us_, 0);
#endif

        ArmDeadline(interrupt_, timeout_us_);
        const int open_ret = avformat_open_input(&format_, source.c_str(), nullptr, &options);
        ArmDeadline(interrupt_, 0);
        if (open_ret < 0) {
            LogError("[FFMPEG] avformat_open_input failed: %s (%d)",
                     AvErrorText(open_ret).c_str(), open_ret);
            av_dict_free(&options);
            Close();
            return false;
        }

        AVDictionaryEntry* unused = nullptr;
        while ((unused = av_dict_get(options, "", unused, AV_DICT_IGNORE_SUFFIX)) != nullptr) {
            LogError("[FFMPEG] unsupported input option: %s", unused->key);
        }
        const bool options_consumed = av_dict_count(options) == 0;
        av_dict_free(&options);
        if (!options_consumed) {
            Close();
            return false;
        }

        ArmDeadline(interrupt_, timeout_us_);
        const int info_ret = avformat_find_stream_info(format_, nullptr);
        ArmDeadline(interrupt_, 0);
        if (info_ret < 0) {
            LogError("[FFMPEG] avformat_find_stream_info failed: %s (%d)",
                     AvErrorText(info_ret).c_str(), info_ret);
            Close();
            return false;
        }

        video_stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_index_ < 0) {
            LogError("[FFMPEG] av_find_best_stream found no video stream: %s (%d)",
                     AvErrorText(video_stream_index_).c_str(), video_stream_index_);
            Close();
            return false;
        }

        stream_ = format_->streams[video_stream_index_];
        const auto* codec = stream_->codecpar;
        if (codec == nullptr || codec->codec_id != AV_CODEC_ID_H264) {
            LogError("[FFMPEG] only H.264 is supported, codec_id=%d",
                     codec == nullptr ? -1 : static_cast<int>(codec->codec_id));
            Close();
            return false;
        }
        if (codec->width != kSourceWidth || codec->height != kSourceHeight) {
            LogError("[FFMPEG] source must be %dx%d, actual=%dx%d",
                     kSourceWidth, kSourceHeight, codec->width, codec->height);
            Close();
            return false;
        }

        const char* demuxer_name = format_->iformat == nullptr ? nullptr : format_->iformat->name;
        if (demuxer_name == nullptr || std::strcmp(demuxer_name, "rtsp") != 0) {
            LogError("[FFMPEG] input is not handled by the RTSP demuxer: %s",
                     demuxer_name == nullptr ? "<null>" : demuxer_name);
            Close();
            return false;
        }

        packet_ = av_packet_alloc();
        filtered_packet_ = av_packet_alloc();
        if (packet_ == nullptr || filtered_packet_ == nullptr) {
            LogError("[FFMPEG] av_packet_alloc failed");
            Close();
            return false;
        }

        if (codec->extradata != nullptr && codec->extradata_size > 0) {
            const auto* extra = codec->extradata;
            const auto extra_size = static_cast<std::size_t>(codec->extradata_size);
            if (HasAnnexBPrefix(extra, extra_size)) {
                annex_b_parameter_sets_.assign(extra, extra + extra_size);
                bitstream_mode_ = "annex-b (RTSP/SDP)";
            } else if (codec->extradata_size >= 7 && codec->extradata[0] == 1) {
                if (!InitializeMp4ToAnnexB()) {
                    Close();
                    return false;
                }
                bitstream_mode_ = "avcC -> conditional h264_mp4toannexb";
            } else {
                LogError("[FFMPEG] H.264 extradata is neither Annex-B nor valid avcC candidate");
                Close();
                return false;
            }
        } else {
            bitstream_mode_ = "annex-b (no SDP parameter sets)";
        }

        const AVRational guessed_rate = av_guess_frame_rate(format_, stream_, nullptr);
        reported_fps_ = guessed_rate.num > 0 && guessed_rate.den > 0 ? av_q2d(guessed_rate) : 0.0;
        LogInfo("[FFMPEG] opened RTSP over TCP: stream=%d codec=H264 size=%dx%d fps=%.3f time_base=%d/%d",
                video_stream_index_, codec->width, codec->height, reported_fps_, stream_->time_base.num,
                stream_->time_base.den);
        LogInfo("[FFMPEG] bitstream mode: %s; explicit decoder APIs: disabled",
                bitstream_mode_.c_str());
        return true;
    }

    ReadResult Read(AccessUnit* access_unit, std::int64_t timeout_us) {
        if (access_unit == nullptr || format_ == nullptr || stream_ == nullptr) {
            LogError("[FFMPEG] Read called with invalid state: access_unit=%p format=%p stream=%p",
                     static_cast<void*>(access_unit), static_cast<void*>(format_),
                     static_cast<void*>(stream_));
            return ReadResult::kError;
        }
        const std::int64_t deadline_us =
            timeout_us > 0 ? av_gettime_relative() + timeout_us : 0;

        while (true) {
            if (interrupt_ != nullptr && interrupt_->stop.load(std::memory_order_relaxed)) {
                return ReadResult::kInterrupted;
            }
            if (deadline_us != 0 && av_gettime_relative() >= deadline_us) {
                return ReadResult::kInterrupted;
            }
            AVPacket* output = nullptr;
            AVRational output_time_base = stream_->time_base;

            if (bsf_ != nullptr) {
                av_packet_unref(filtered_packet_);
                const int receive_ret = av_bsf_receive_packet(bsf_, filtered_packet_);
                if (receive_ret == 0) {
                    output = filtered_packet_;
                    output_time_base = bsf_->time_base_out;
                } else if (receive_ret == AVERROR_EOF) {
                    return ReadResult::kEof;
                } else if (receive_ret != AVERROR(EAGAIN)) {
                    ++ffmpeg_errors_;
                    LogError("[FFMPEG] av_bsf_receive_packet failed: %s (%d)",
                             AvErrorText(receive_ret).c_str(), receive_ret);
                    return ReadResult::kError;
                }
            }

            if (output == nullptr) {
                const ReadResult input_result = ReadSelectedPacket(deadline_us);
                if (input_result != ReadResult::kPacket) {
                    if (input_result == ReadResult::kEof && bsf_ != nullptr && !bsf_eof_sent_) {
                        const int flush_ret = av_bsf_send_packet(bsf_, nullptr);
                        if (flush_ret < 0 && flush_ret != AVERROR_EOF) {
                            ++ffmpeg_errors_;
                            LogError("[FFMPEG] flush BSF failed: %s (%d)",
                                     AvErrorText(flush_ret).c_str(), flush_ret);
                            return ReadResult::kError;
                        }
                        bsf_eof_sent_ = true;
                        continue;
                    }
                    return input_result;
                }

                if (bsf_ == nullptr) {
                    output = packet_;
                } else {
                    const int send_ret = av_bsf_send_packet(bsf_, packet_);
                    if (send_ret < 0) {
                        ++ffmpeg_errors_;
                        LogError("[FFMPEG] av_bsf_send_packet failed: %s (%d)",
                                 AvErrorText(send_ret).c_str(), send_ret);
                        av_packet_unref(packet_);
                        return ReadResult::kError;
                    }
                    continue;
                }
            }

            const bool accepted = BuildAccessUnit(*output, output_time_base, access_unit);
            av_packet_unref(output);
            if (accepted) {
                return ReadResult::kPacket;
            }
            if (fatal_packet_error_) {
                return ReadResult::kError;
            }
        }
    }

    void Close() {
        if (packet_ != nullptr) {
            av_packet_free(&packet_);
        }
        if (filtered_packet_ != nullptr) {
            av_packet_free(&filtered_packet_);
        }
        if (bsf_ != nullptr) {
            av_bsf_free(&bsf_);
        }
        if (format_ != nullptr) {
            avformat_close_input(&format_);
        }
        stream_ = nullptr;
        video_stream_index_ = -1;
        bsf_eof_sent_ = false;
    }

    std::uint64_t input_packets() const { return input_packets_; }
    std::uint64_t skipped_before_idr() const { return skipped_before_idr_; }
    std::uint64_t ffmpeg_errors() const { return ffmpeg_errors_; }

private:
    bool InitializeMp4ToAnnexB() {
        const AVBitStreamFilter* filter = av_bsf_get_by_name("h264_mp4toannexb");
        if (filter == nullptr) {
            LogError("[FFMPEG] bundled FFmpeg has no h264_mp4toannexb BSF");
            return false;
        }
        int ret = av_bsf_alloc(filter, &bsf_);
        if (ret < 0) {
            LogError("[FFMPEG] av_bsf_alloc failed: %s (%d)", AvErrorText(ret).c_str(), ret);
            return false;
        }
        ret = avcodec_parameters_copy(bsf_->par_in, stream_->codecpar);
        if (ret < 0) {
            LogError("[FFMPEG] avcodec_parameters_copy failed: %s (%d)",
                     AvErrorText(ret).c_str(), ret);
            return false;
        }
        bsf_->time_base_in = stream_->time_base;
        ret = av_bsf_init(bsf_);
        if (ret < 0) {
            LogError("[FFMPEG] av_bsf_init failed: %s (%d)", AvErrorText(ret).c_str(), ret);
            return false;
        }
        return true;
    }

    ReadResult ReadSelectedPacket(std::int64_t deadline_us) {
        while (true) {
            if (interrupt_ != nullptr && interrupt_->stop.load(std::memory_order_relaxed)) {
                return ReadResult::kInterrupted;
            }
            if (deadline_us != 0 && av_gettime_relative() >= deadline_us) {
                return ReadResult::kInterrupted;
            }
            av_packet_unref(packet_);
            if (interrupt_ != nullptr) {
                interrupt_->deadline_us.store(deadline_us, std::memory_order_relaxed);
            }
            const int ret = av_read_frame(format_, packet_);
            ArmDeadline(interrupt_, 0);
            if (ret == AVERROR_EOF) {
                return ReadResult::kEof;
            }
            if (ret == AVERROR_EXIT || (interrupt_ != nullptr && interrupt_->stop.load(std::memory_order_relaxed))) {
                return ReadResult::kInterrupted;
            }
            if (ret < 0) {
                ++ffmpeg_errors_;
                LogError("[FFMPEG] av_read_frame failed: %s (%d)", AvErrorText(ret).c_str(), ret);
                return ReadResult::kError;
            }
            if (interrupt_ != nullptr && interrupt_->stop.load(std::memory_order_relaxed)) {
                av_packet_unref(packet_);
                return ReadResult::kInterrupted;
            }
            if (packet_->stream_index != video_stream_index_) {
                continue;
            }
            ++input_packets_;
            if ((packet_->flags & AV_PKT_FLAG_CORRUPT) != 0) {
                ++ffmpeg_errors_;
                waiting_for_idr_ = true;
                pre_idr_parameter_sets_.clear();
                LogWarning("[FFMPEG] discard packet marked corrupt, packet=%llu",
                           static_cast<unsigned long long>(input_packets_));
                continue;
            }
            return ReadResult::kPacket;
        }
    }

    bool BuildAccessUnit(const AVPacket& packet, AVRational time_base, AccessUnit* access_unit) {
        if (packet.data == nullptr || packet.size <= 0 || access_unit == nullptr) {
            ++ffmpeg_errors_;
            LogError("[FFMPEG] invalid demuxed packet: data=%p size=%d access_unit=%p",
                     static_cast<void*>(packet.data), packet.size, static_cast<void*>(access_unit));
            return false;
        }
        if (!HasAnnexBPrefix(packet.data, static_cast<std::size_t>(packet.size))) {
            ++ffmpeg_errors_;
            fatal_packet_error_ = true;
            LogError("[FFMPEG] demuxed H.264 packet is not Annex-B; refusing to guess packet format");
            return false;
        }

        const auto summary = InspectAnnexBNals(packet.data, static_cast<std::size_t>(packet.size));

        if (waiting_for_idr_ && !summary.has_idr) {
            if ((summary.has_sps || summary.has_pps) && !summary.has_vcl) {
                constexpr std::size_t kMaxParameterSetBytes = 1024U * 1024U;
                const auto packet_size = static_cast<std::size_t>(packet.size);
                if (pre_idr_parameter_sets_.size() + packet_size > kMaxParameterSetBytes) {
                    ++ffmpeg_errors_;
                    fatal_packet_error_ = true;
                    LogError("[FFMPEG] pre-IDR SPS/PPS cache exceeds %zu bytes", kMaxParameterSetBytes);
                    return false;
                }
                pre_idr_parameter_sets_.insert(pre_idr_parameter_sets_.end(),
                                               packet.data, packet.data + packet.size);
            }
            ++skipped_before_idr_;
            return false;
        }

        const std::vector<std::uint8_t>* prefix = nullptr;
        if (summary.has_idr && !(summary.has_sps && summary.has_pps)) {
            if (!annex_b_parameter_sets_.empty()) {
                prefix = &annex_b_parameter_sets_;
            } else if (!pre_idr_parameter_sets_.empty()) {
                prefix = &pre_idr_parameter_sets_;
            }
        }

        access_unit->data.clear();
        const std::size_t prefix_size = prefix == nullptr ? 0 : prefix->size();
        access_unit->data.reserve(prefix_size + static_cast<std::size_t>(packet.size));
        if (prefix != nullptr) {
            access_unit->data.insert(access_unit->data.end(), prefix->begin(), prefix->end());
        }
        access_unit->data.insert(access_unit->data.end(), packet.data, packet.data + packet.size);

        if (waiting_for_idr_ && summary.has_idr) {
            waiting_for_idr_ = false;
            if (prefix == nullptr && !(summary.has_sps && summary.has_pps)) {
                LogWarning("[FFMPEG] first IDR has no observable SPS/PPS; VDEC may reject the stream");
            }
        }

        const std::int64_t packet_ts = packet.pts != AV_NOPTS_VALUE ? packet.pts : packet.dts;
        std::uint64_t duration_us = 0;
        if (packet.duration > 0) {
            const auto converted = av_rescale_q(packet.duration, time_base, AV_TIME_BASE_Q);
            duration_us = converted > 0 ? static_cast<std::uint64_t>(converted) : 0;
        }
        if (duration_us == 0 && reported_fps_ > 0.0) {
            duration_us = static_cast<std::uint64_t>(1000000.0 / reported_fps_);
        }
        if (duration_us == 0) {
            duration_us = 33333;
        }

        if (packet_ts != AV_NOPTS_VALUE) {
            const auto converted = av_rescale_q(packet_ts, time_base, AV_TIME_BASE_Q);
            access_unit->pts_us = converted >= 0 ? static_cast<std::uint64_t>(converted) : 0;
            synthetic_pts_us_ = std::max(synthetic_pts_us_, access_unit->pts_us + duration_us);
        } else {
            access_unit->pts_us = synthetic_pts_us_;
            synthetic_pts_us_ += duration_us;
        }
        return true;
    }

    InterruptState* interrupt_{nullptr};
    AVFormatContext* format_{nullptr};
    AVStream* stream_{nullptr};
    AVBSFContext* bsf_{nullptr};
    AVPacket* packet_{nullptr};
    AVPacket* filtered_packet_{nullptr};
    int video_stream_index_{-1};
    std::int64_t timeout_us_{0};
    std::string bitstream_mode_;
    std::vector<std::uint8_t> annex_b_parameter_sets_;
    std::vector<std::uint8_t> pre_idr_parameter_sets_;
    std::uint64_t input_packets_{0};
    std::uint64_t skipped_before_idr_{0};
    std::uint64_t ffmpeg_errors_{0};
    std::uint64_t synthetic_pts_us_{0};
    double reported_fps_{0.0};
    bool waiting_for_idr_{true};
    bool bsf_eof_sent_{false};
    bool fatal_packet_error_{false};
};

bool RecordCleanupResult(const char* api, AX_S32 result) {
    if (result == AX_SUCCESS) {
        return true;
    }
    LogError("[CLEANUP] %s failed: 0x%08X", api, static_cast<unsigned int>(result));
    return false;
}

class AxclEnvironment final {
public:
    ~AxclEnvironment() {
        (void)Shutdown();
    }

    bool Initialize(const Options& options) {
        (void)Shutdown();
        if (const auto ret = axclInit(options.axcl_config.empty() ? nullptr : options.axcl_config.c_str());
            ret != AXCL_SUCC) {
            LogError("[AXCL] axclInit failed: 0x%08X", static_cast<unsigned int>(ret));
            return false;
        }
        axcl_initialized_ = true;

        std::int32_t version_major = 0;
        std::int32_t version_minor = 0;
        std::int32_t version_patch = 0;
        if (const auto ret = axclrtGetVersion(&version_major, &version_minor, &version_patch);
            ret == AXCL_SUCC) {
            LogInfo("[AXCL] runtime version=%d.%d.%d", version_major, version_minor, version_patch);
        } else {
            LogWarning("[AXCL] axclrtGetVersion failed: 0x%08X", static_cast<unsigned int>(ret));
        }

        axclrtDeviceList devices{};
        if (const auto ret = axclrtGetDeviceList(&devices); ret != AXCL_SUCC || devices.num == 0) {
            LogError("[AXCL] axclrtGetDeviceList failed: 0x%08X, devices=%u",
                     static_cast<unsigned int>(ret), devices.num);
            (void)Shutdown();
            return false;
        }
        if (options.device_index < 0 || options.device_index >= static_cast<int>(devices.num)) {
            LogError("[AXCL] device index %d is outside [0, %u)", options.device_index, devices.num);
            (void)Shutdown();
            return false;
        }
        runtime_device_id_ = devices.devices[options.device_index];

        if (const auto ret = axclrtSetDevice(runtime_device_id_); ret != AXCL_SUCC) {
            LogError("[AXCL] axclrtSetDevice(%d) failed: 0x%08X",
                     runtime_device_id_, static_cast<unsigned int>(ret));
            (void)Shutdown();
            return false;
        }
        device_set_ = true;

        if (const auto ret = axclrtCreateContext(&context_, runtime_device_id_);
            ret != AXCL_SUCC || context_ == nullptr) {
            LogError("[AXCL] axclrtCreateContext failed: 0x%08X", static_cast<unsigned int>(ret));
            (void)Shutdown();
            return false;
        }
        if (!EnsureCurrentContext()) {
            (void)Shutdown();
            return false;
        }

        if (const auto ret = AXCL_SYS_Init(); ret != AX_SUCCESS) {
            LogError("[AXCL] AXCL_SYS_Init failed: 0x%08X", static_cast<unsigned int>(ret));
            (void)Shutdown();
            return false;
        }
        sys_initialized_ = true;

        AX_VDEC_MOD_ATTR_T vdec_attr{};
        vdec_attr.u32MaxGroupCount = 1;
        if (const auto ret = AXCL_VDEC_Init(&vdec_attr); ret != AX_SUCCESS) {
            LogError("[AXCL] AXCL_VDEC_Init failed: 0x%08X", static_cast<unsigned int>(ret));
            (void)Shutdown();
            return false;
        }
        vdec_initialized_ = true;

        if (options.mode != RunMode::kVdecSmoke) {
            if (const auto ret = AXCL_IVPS_Init(); ret != AX_SUCCESS) {
                LogError("[AXCL] AXCL_IVPS_Init failed: 0x%08X", static_cast<unsigned int>(ret));
                (void)Shutdown();
                return false;
            }
            ivps_initialized_ = true;
        }

        if (options.mode == RunMode::kInfer) {
            if (const auto ret = axclrtEngineInit(AXCL_VNPU_DISABLE); ret != AXCL_SUCC) {
                LogError("[AXCL] axclrtEngineInit failed: 0x%08X", static_cast<unsigned int>(ret));
                (void)Shutdown();
                return false;
            }
            engine_initialized_ = true;
        }

        LogInfo("[AXCL] initialized device_index=%d runtime_device=%d context=%p",
                options.device_index, runtime_device_id_, context_);
        return true;
    }

    bool EnsureCurrentContext() const {
        if (context_ == nullptr) {
            LogError("[AXCL] cannot bind a null context");
            return false;
        }
        const auto ret = axclrtSetCurrentContext(context_);
        if (ret != AXCL_SUCC) {
            LogError("[AXCL] axclrtSetCurrentContext failed: 0x%08X", static_cast<unsigned int>(ret));
            return false;
        }
        return true;
    }

    bool Shutdown() {
        bool successful = true;
        if (context_ != nullptr) {
            successful = RecordCleanupResult("axclrtSetCurrentContext",
                                             axclrtSetCurrentContext(context_)) && successful;
        }
        if (engine_initialized_) {
            successful = RecordCleanupResult("axclrtEngineFinalize", axclrtEngineFinalize()) && successful;
            engine_initialized_ = false;
        }
        if (ivps_initialized_) {
            successful = RecordCleanupResult("AXCL_IVPS_Deinit", AXCL_IVPS_Deinit()) && successful;
            ivps_initialized_ = false;
        }
        if (vdec_initialized_) {
            successful = RecordCleanupResult("AXCL_VDEC_Deinit", AXCL_VDEC_Deinit()) && successful;
            vdec_initialized_ = false;
        }
        if (sys_initialized_) {
            successful = RecordCleanupResult("AXCL_SYS_Deinit", AXCL_SYS_Deinit()) && successful;
            sys_initialized_ = false;
        }
        if (context_ != nullptr) {
            successful = RecordCleanupResult("axclrtDestroyContext",
                                             axclrtDestroyContext(context_)) && successful;
            context_ = nullptr;
        }
        if (device_set_) {
            successful = RecordCleanupResult("axclrtResetDevice",
                                             axclrtResetDevice(runtime_device_id_)) && successful;
            device_set_ = false;
        }
        if (axcl_initialized_) {
            successful = RecordCleanupResult("axclFinalize", axclFinalize()) && successful;
            axcl_initialized_ = false;
        }
        runtime_device_id_ = -1;
        return successful;
    }

private:
    axclrtContext context_{nullptr};
    int runtime_device_id_{-1};
    bool axcl_initialized_{false};
    bool device_set_{false};
    bool sys_initialized_{false};
    bool vdec_initialized_{false};
    bool ivps_initialized_{false};
    bool engine_initialized_{false};
};

struct VdecStatistics {
    std::uint64_t attempted_access_units{0};
    std::uint64_t sent_access_units{0};
    std::uint64_t decoded_frames{0};
    std::uint64_t send_calls{0};
    std::uint64_t send_failures{0};
    std::uint64_t send_runtime_timeouts{0};
    std::uint64_t slow_send_calls{0};
    std::uint64_t send_full_retries{0};
    std::uint64_t failure_events{0};
    std::uint64_t errors{0};
    std::uint64_t hardware_decode_errors{0};
    double send_total_ms{0.0};
    double send_max_ms{0.0};
    AX_U32 left_stream_frames{0};
    AX_U32 left_output_frames{0};
    std::uint64_t last_pts_us{0};
    AX_U32 last_width{0};
    AX_U32 last_height{0};
    AX_IMG_FORMAT_E last_format{AX_FORMAT_INVALID};
};

class NativeVdec final {
public:
    using FrameHandler = std::function<bool(const AX_VIDEO_FRAME_INFO_T&)>;

    explicit NativeVdec(const std::atomic<bool>* stop_requested) : stop_requested_(stop_requested) {}

    ~NativeVdec() {
        (void)Close();
    }

    bool Open(int width, int height) {
        (void)Close();
        const AX_U32 aligned_width = AlignUp(static_cast<AX_U32>(width), 16);
        const AX_U32 aligned_height = AlignUp(static_cast<AX_U32>(height), 16);

        AX_VDEC_GRP_ATTR_T group_attr{};
        group_attr.enCodecType = PT_H264;
        group_attr.enInputMode = AX_VDEC_INPUT_MODE_FRAME;
        group_attr.u32MaxPicWidth = aligned_width;
        group_attr.u32MaxPicHeight = aligned_height;
        group_attr.u32StreamBufSize = std::max<AX_U32>(aligned_width * aligned_height * 2U, 1024U * 1024U);
        group_attr.bSdkAutoFramePool = AX_TRUE;

        if (const auto ret = AXCL_VDEC_CreateGrpEx(&group_, &group_attr); ret != AX_SUCCESS) {
            group_ = -1;
            LogError("[VDEC] AXCL_VDEC_CreateGrpEx failed: 0x%08X", static_cast<unsigned int>(ret));
            ++statistics_.errors;
            return false;
        }

        AX_VDEC_GRP_PARAM_T group_param{};
        group_param.stVdecVideoParam.enOutputOrder = AX_VDEC_OUTPUT_ORDER_DISP;
        group_param.stVdecVideoParam.enVdecMode = VIDEO_DEC_MODE_IPB;
        if (const auto ret = AXCL_VDEC_SetGrpParam(group_, &group_param); ret != AX_SUCCESS) {
            LogError("[VDEC] AXCL_VDEC_SetGrpParam failed: 0x%08X", static_cast<unsigned int>(ret));
            ++statistics_.errors;
            (void)Close();
            return false;
        }

        AX_VDEC_CHN_ATTR_T channel_attr{};
        channel_attr.u32PicWidth = static_cast<AX_U32>(width);
        channel_attr.u32PicHeight = static_cast<AX_U32>(height);
        channel_attr.u32FrameStride = AlignUp(channel_attr.u32PicWidth, 256);
        channel_attr.u32OutputFifoDepth = 3;
        channel_attr.u32FrameBufCnt = kH264FrameBufferCount;
        channel_attr.enOutputMode = AX_VDEC_OUTPUT_ORIGINAL;
        channel_attr.enImgFormat = AX_FORMAT_YUV420_SEMIPLANAR;
        channel_attr.stCompressInfo.enCompressMode = AX_COMPRESS_MODE_NONE;
        channel_attr.u32FrameBufSize = AX_VDEC_GetPicBufferSize(
            channel_attr.u32PicWidth, channel_attr.u32PicHeight, channel_attr.enImgFormat,
            &channel_attr.stCompressInfo, PT_H264);
        if (channel_attr.u32FrameBufSize == 0) {
            LogError("[VDEC] AX_VDEC_GetPicBufferSize returned 0");
            ++statistics_.errors;
            (void)Close();
            return false;
        }
        if (const auto ret = AXCL_VDEC_SetChnAttr(group_, kVdecChannel, &channel_attr); ret != AX_SUCCESS) {
            LogError("[VDEC] AXCL_VDEC_SetChnAttr failed: 0x%08X", static_cast<unsigned int>(ret));
            ++statistics_.errors;
            (void)Close();
            return false;
        }
        if (const auto ret = AXCL_VDEC_EnableChn(group_, kVdecChannel); ret != AX_SUCCESS) {
            LogError("[VDEC] AXCL_VDEC_EnableChn failed: 0x%08X", static_cast<unsigned int>(ret));
            ++statistics_.errors;
            (void)Close();
            return false;
        }
        channel_enabled_ = true;

        if (const auto ret = AXCL_VDEC_SetDisplayMode(group_, AX_VDEC_DISPLAY_MODE_PLAYBACK);
            ret != AX_SUCCESS) {
            LogError("[VDEC] AXCL_VDEC_SetDisplayMode failed: 0x%08X", static_cast<unsigned int>(ret));
            ++statistics_.errors;
            (void)Close();
            return false;
        }
        AX_VDEC_RECV_PIC_PARAM_T receive_param{};
        receive_param.s32RecvPicNum = -1;
        if (const auto ret = AXCL_VDEC_StartRecvStream(group_, &receive_param); ret != AX_SUCCESS) {
            LogError("[VDEC] AXCL_VDEC_StartRecvStream failed: 0x%08X", static_cast<unsigned int>(ret));
            ++statistics_.errors;
            (void)Close();
            return false;
        }
        started_ = true;
        LogInfo("[VDEC] group=%d input_mode=FRAME output=NV12 original=%dx%d stride=%u buffers=%u "
                "send_wait_ms=%d slow_send_ms=%.1f",
                group_, width, height, channel_attr.u32FrameStride, channel_attr.u32FrameBufCnt,
                kAxWaitMs, kSlowVdecSendMilliseconds);
        return true;
    }

    bool Send(const AccessUnit& access_unit, const FrameHandler& handler) {
        const std::uint64_t access_unit_sequence = ++statistics_.attempted_access_units;
        if (group_ < 0 || access_unit.data.empty() ||
            access_unit.data.size() > static_cast<std::size_t>(std::numeric_limits<AX_U32>::max())) {
            ++statistics_.errors;
            LogError("[VDEC] invalid access unit: au_seq=%llu group=%d bytes=%zu",
                     static_cast<unsigned long long>(access_unit_sequence), group_, access_unit.data.size());
            return false;
        }

        AX_VDEC_STREAM_T stream{};
        stream.u64PTS = access_unit.pts_us;
        stream.bEndOfFrame = AX_TRUE;
        stream.bEndOfStream = AX_FALSE;
        stream.bSkipDisplay = AX_FALSE;
        stream.u32StreamPackLen = static_cast<AX_U32>(access_unit.data.size());
        stream.pu8Addr = const_cast<AX_U8*>(access_unit.data.data());
        stream.u64PhyAddr = 0;

        std::uint64_t retry_index = 0;
        while (!StopRequested()) {
            const SendCallResult call = CallSendStream(stream, "data", access_unit_sequence, retry_index);
            if (call.result == AX_SUCCESS) {
                ++statistics_.sent_access_units;
                return DrainAvailable(handler, 0, nullptr);
            }
            if (!IsBufferPressure(call.result)) {
                ++statistics_.errors;
                RecordSendFailure(call, stream, "data", access_unit_sequence, retry_index);
                return false;
            }

            ++statistics_.send_full_retries;
            ++retry_index;
            bool received = false;
            if (!DrainAvailable(handler, kAxWaitMs, &received)) {
                return false;
            }
            if (!received) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        return false;
    }

    bool Finish(const FrameHandler& handler) {
        if (group_ < 0 || !started_) {
            return true;
        }

        const auto send_deadline = Clock::now() + std::chrono::seconds(10);
        AX_VDEC_STREAM_T stream{};
        stream.bEndOfFrame = AX_TRUE;
        stream.bEndOfStream = AX_TRUE;
        bool eos_sent = false;
        std::uint64_t retry_index = 0;
        while (Clock::now() < send_deadline) {
            const SendCallResult call = CallSendStream(stream, "eos", 0, retry_index);
            if (call.result == AX_SUCCESS || call.result == AX_ERR_VDEC_FLOW_END) {
                eos_sent = true;
                break;
            }
            if (!IsBufferPressure(call.result)) {
                ++statistics_.errors;
                RecordSendFailure(call, stream, "eos", 0, retry_index);
                return false;
            }
            ++statistics_.send_full_retries;
            ++retry_index;
            if (!DrainAvailable(handler, kAxWaitMs, nullptr)) {
                return false;
            }
        }
        if (!eos_sent) {
            ++statistics_.errors;
            LogError("[VDEC] EOS send timed out");
            return false;
        }

        const auto drain_deadline = Clock::now() + std::chrono::seconds(10);
        while (Clock::now() < drain_deadline) {
            bool flow_end = false;
            if (!DrainAvailable(handler, kAxWaitMs, nullptr, &flow_end)) {
                return false;
            }
            if (flow_end) {
                return true;
            }
        }
        ++statistics_.errors;
        LogError("[VDEC] EOS drain timed out before AX_ERR_VDEC_FLOW_END");
        return false;
    }

    bool Close() {
        bool successful = true;
        if (group_ >= 0 && started_) {
            successful = RecordCleanupResult("AXCL_VDEC_StopRecvStream",
                                             AXCL_VDEC_StopRecvStream(group_)) && successful;
            successful = RecordCleanupResult("AXCL_VDEC_ResetGrp",
                                             AXCL_VDEC_ResetGrp(group_)) && successful;
            started_ = false;
        }
        if (group_ >= 0 && channel_enabled_) {
            successful = RecordCleanupResult("AXCL_VDEC_DisableChn",
                                             AXCL_VDEC_DisableChn(group_, kVdecChannel)) && successful;
            channel_enabled_ = false;
        }
        if (group_ >= 0) {
            successful = RecordCleanupResult("AXCL_VDEC_DestroyGrp",
                                             AXCL_VDEC_DestroyGrp(group_)) && successful;
            group_ = -1;
        }
        return successful;
    }

    const VdecStatistics& statistics() const { return statistics_; }

    bool RefreshStatus() {
        if (group_ < 0) {
            return false;
        }
        AX_VDEC_GRP_STATUS_T status{};
        const auto ret = AXCL_VDEC_QueryStatus(group_, &status);
        if (ret != AX_SUCCESS) {
            ++statistics_.errors;
            LogError("[VDEC] AXCL_VDEC_QueryStatus failed: 0x%08X", static_cast<unsigned int>(ret));
            return false;
        }
        UpdateStatusStatistics(status);
        return true;
    }

private:
    struct SendCallResult {
        AX_S32 result{AX_SUCCESS};
        std::uint64_t call_sequence{0};
        double elapsed_ms{0.0};
    };

    static bool IsBufferPressure(AX_S32 result) {
        return result == AX_ERR_VDEC_BUF_FULL || result == AX_ERR_VDEC_QUEUE_FULL;
    }

    SendCallResult CallSendStream(const AX_VDEC_STREAM_T& stream, const char* kind,
                                  std::uint64_t access_unit_sequence, std::uint64_t retry_index) {
        SendCallResult call{};
        call.call_sequence = ++statistics_.send_calls;
        const auto begin = Clock::now();
        call.result = AXCL_VDEC_SendStream(group_, &stream, kAxWaitMs);
        call.elapsed_ms = ElapsedMilliseconds(begin);
        statistics_.send_total_ms += call.elapsed_ms;
        statistics_.send_max_ms = std::max(statistics_.send_max_ms, call.elapsed_ms);

        if (call.elapsed_ms >= kSlowVdecSendMilliseconds &&
            (call.result == AX_SUCCESS || call.result == AX_ERR_VDEC_FLOW_END ||
             IsBufferPressure(call.result))) {
            ++statistics_.slow_send_calls;
            LogWarning("[VDEC] slow SendStream: kind=%s call_seq=%llu au_seq=%llu retry=%llu "
                       "requested_wait_ms=%d call_ms=%.3f ret=0x%08X pts=%llu bytes=%u",
                       kind, static_cast<unsigned long long>(call.call_sequence),
                       static_cast<unsigned long long>(access_unit_sequence),
                       static_cast<unsigned long long>(retry_index), kAxWaitMs, call.elapsed_ms,
                       static_cast<unsigned int>(call.result),
                       static_cast<unsigned long long>(stream.u64PTS), stream.u32StreamPackLen);
        }
        return call;
    }

    void RecordSendFailure(const SendCallResult& call, const AX_VDEC_STREAM_T& stream,
                           const char* kind, std::uint64_t access_unit_sequence,
                           std::uint64_t retry_index) {
        ++statistics_.send_failures;
        if (call.elapsed_ms >= kSlowVdecSendMilliseconds) {
            ++statistics_.slow_send_calls;
        }
        const bool runtime_task_timeout = call.result == kAxclRuntimeTaskTimeout;
        if (runtime_task_timeout) {
            ++statistics_.send_runtime_timeouts;
        }
        const std::uint64_t failure_event_id = ++statistics_.failure_events;
        const std::uint32_t code = static_cast<std::uint32_t>(call.result);
        LogError("[VDEC] SendStream failed: event=%llu kind=%s call_seq=%llu au_seq=%llu retry=%llu "
                 "requested_wait_ms=%d call_ms=%.3f ret=0x%08X module=0x%02X ax_id=0x%02X "
                 "submodule=0x%02X reason=0x%02X classification=%s pts=%llu bytes=%u "
                 "sent_au=%llu decoded_frames=%llu full_retries=%llu",
                 static_cast<unsigned long long>(failure_event_id), kind,
                 static_cast<unsigned long long>(call.call_sequence),
                 static_cast<unsigned long long>(access_unit_sequence),
                 static_cast<unsigned long long>(retry_index), kAxWaitMs, call.elapsed_ms, code,
                 static_cast<unsigned int>((code >> 24U) & 0x7FU),
                 static_cast<unsigned int>((code >> 16U) & 0xFFU),
                 static_cast<unsigned int>((code >> 8U) & 0xFFU),
                 static_cast<unsigned int>(code & 0xFFU),
                 runtime_task_timeout ? "axcl-runtime-task-timeout" : "other",
                 static_cast<unsigned long long>(stream.u64PTS), stream.u32StreamPackLen,
                 static_cast<unsigned long long>(statistics_.sent_access_units),
                 static_cast<unsigned long long>(statistics_.decoded_frames),
                 static_cast<unsigned long long>(statistics_.send_full_retries));
        LogFailureStatusSnapshot(failure_event_id);
    }

    void LogFailureStatusSnapshot(std::uint64_t failure_event_id) {
        AX_VDEC_GRP_STATUS_T status{};
        const auto begin = Clock::now();
        const AX_S32 result = AXCL_VDEC_QueryStatus(group_, &status);
        const double query_ms = ElapsedMilliseconds(begin);
        if (result != AX_SUCCESS) {
            LogError("[VDEC] fault QueryStatus failed: event=%llu query_ms=%.3f ret=0x%08X",
                     static_cast<unsigned long long>(failure_event_id), query_ms,
                     static_cast<unsigned int>(result));
            return;
        }

        UpdateStatusStatistics(status);
        const auto& error = status.stVdecDecErr;
        LogInfoFlush("[VDEC][FAULT_STATUS] event=%llu query_ms=%.3f left_bytes=%u left_frames=%u "
                     "left_pics=%u started=%d recv_frames=%u decoded_stream_frames=%u size=%ux%u "
                     "input_fifo_full=%d format_err=%d pic_size_err=%d stream_unsupported=%d "
                     "pack_err=%d ref_err=%d pic_buf_size_err=%d stream_size_over=%d "
                     "stream_not_release=%d",
                     static_cast<unsigned long long>(failure_event_id), query_ms,
                     status.u32LeftStreamBytes, status.u32LeftStreamFrames,
                     status.u32LeftPics[kVdecChannel], static_cast<int>(status.bStartRecvStream),
                     status.u32RecvStreamFrames, status.u32DecodeStreamFrames,
                     status.u32PicWidth, status.u32PicHeight, static_cast<int>(status.bInputFifoIsFull),
                     error.s32FormatErr, error.s32PicSizeErrSet, error.s32StreamUnsprt,
                     error.s32PackErr, error.s32RefErrSet, error.s32PicBufSizeErrSet,
                     error.s32StreamSizeOver, error.s32VdecStreamNotRelease);
    }

    void UpdateStatusStatistics(const AX_VDEC_GRP_STATUS_T& status) {
        const auto nonnegative = [](AX_S32 value) {
            return value > 0 ? static_cast<std::uint64_t>(value) : 0ULL;
        };
        const auto& error = status.stVdecDecErr;
        statistics_.hardware_decode_errors =
            nonnegative(error.s32FormatErr) + nonnegative(error.s32PicSizeErrSet) +
            nonnegative(error.s32StreamUnsprt) + nonnegative(error.s32PackErr) +
            nonnegative(error.s32RefErrSet) + nonnegative(error.s32PicBufSizeErrSet) +
            nonnegative(error.s32StreamSizeOver) + nonnegative(error.s32VdecStreamNotRelease);
        statistics_.left_stream_frames = status.u32LeftStreamFrames;
        statistics_.left_output_frames = status.u32LeftPics[kVdecChannel];
    }

    bool StopRequested() const {
        return stop_requested_ != nullptr && stop_requested_->load(std::memory_order_relaxed);
    }

    bool DrainAvailable(const FrameHandler& handler, AX_S32 first_wait_ms, bool* received_any,
                        bool* flow_end = nullptr) {
        if (received_any != nullptr) {
            *received_any = false;
        }
        if (flow_end != nullptr) {
            *flow_end = false;
        }

        AX_S32 wait_ms = first_wait_ms;
        while (true) {
            AX_VIDEO_FRAME_INFO_T frame{};
            const auto ret = AXCL_VDEC_GetChnFrame(group_, kVdecChannel, &frame, wait_ms);
            if (ret == AX_ERR_VDEC_FLOW_END) {
                if (flow_end != nullptr) {
                    *flow_end = true;
                }
                return true;
            }
            if (ret != AX_SUCCESS) {
                if (ret == AX_ERR_VDEC_BUF_EMPTY || ret == AX_ERR_VDEC_QUEUE_EMPTY ||
                    ret == AX_ERR_VDEC_TIMED_OUT) {
                    return true;
                }
                ++statistics_.errors;
                LogError("[VDEC] AXCL_VDEC_GetChnFrame failed: 0x%08X",
                         static_cast<unsigned int>(ret));
                return false;
            }

            if (received_any != nullptr) {
                *received_any = true;
            }
            ++statistics_.decoded_frames;
            statistics_.last_pts_us = frame.stVFrame.u64PTS;
            statistics_.last_width = frame.stVFrame.u32Width;
            statistics_.last_height = frame.stVFrame.u32Height;
            statistics_.last_format = frame.stVFrame.enImgFormat;

            bool handled = false;
            try {
                handled = handler(frame);
            } catch (const std::exception& exception) {
                ++statistics_.errors;
                LogError("[VDEC] frame handler exception: %s", exception.what());
            } catch (...) {
                ++statistics_.errors;
                LogError("[VDEC] frame handler threw an unknown exception");
            }
            const auto release_ret = AXCL_VDEC_ReleaseChnFrame(group_, kVdecChannel, &frame);
            if (release_ret != AX_SUCCESS) {
                ++statistics_.errors;
                LogError("[VDEC] AXCL_VDEC_ReleaseChnFrame failed: 0x%08X",
                         static_cast<unsigned int>(release_ret));
                return false;
            }
            if (!handled) {
                return false;
            }
            wait_ms = 0;
        }
    }

    AX_VDEC_GRP group_{-1};
    const std::atomic<bool>* stop_requested_{nullptr};
    bool channel_enabled_{false};
    bool started_{false};
    VdecStatistics statistics_{};
};

struct IvpsStatistics {
    std::uint64_t frames{0};
    std::uint64_t errors{0};
    double total_ms{0.0};
};

class NativeIvpsPreprocessor final {
public:
    ~NativeIvpsPreprocessor() {
        (void)Close();
    }

    bool Open() {
        (void)Close();
        const auto* token = reinterpret_cast<const AX_S8*>("yolo26-ivps-input");
        if (const auto ret = AXCL_SYS_MemAlloc(&physical_address_, &virtual_address_,
                                               static_cast<AX_U32>(kInputBytes), 0x1000, token);
            ret != AX_SUCCESS || physical_address_ == 0 || virtual_address_ == nullptr) {
            LogError("[IVPS] AXCL_SYS_MemAlloc failed: 0x%08X", static_cast<unsigned int>(ret));
            physical_address_ = 0;
            virtual_address_ = nullptr;
            ++statistics_.errors;
            return false;
        }

        output_frame_ = {};
        output_frame_.u32Width = kInputWidth;
        output_frame_.u32Height = kInputHeight;
        output_frame_.s16CropX = 0;
        output_frame_.s16CropY = 0;
        output_frame_.s16CropWidth = kInputWidth;
        output_frame_.s16CropHeight = kInputHeight;
        output_frame_.enImgFormat = AX_FORMAT_BGR888;
        output_frame_.enVscanFormat = AX_VSCAN_FORMAT_RASTER;
        output_frame_.stCompressInfo.enCompressMode = AX_COMPRESS_MODE_NONE;
        output_frame_.stDynamicRange = AX_DYNAMIC_RANGE_SDR8;
        output_frame_.stColorGamut = AX_COLOR_GAMUT_BT709;
        output_frame_.u32PicStride[0] = static_cast<AX_U32>(kInputStride);
        output_frame_.u32FrameSize = static_cast<AX_U32>(kInputBytes);
        output_frame_.u64PhyAddr[0] = physical_address_;
        output_frame_.u64VirAddr[0] = static_cast<AX_U64>(reinterpret_cast<std::uintptr_t>(virtual_address_));
        for (auto& block_id : output_frame_.u32BlkId) {
            block_id = AX_INVALID_BLOCKID;
        }

        LogInfo("[IVPS] output=640x640 BGR888 NHWC/U8 stride=%zu bytes=%zu "
                "aspect=AUTO center bg=0 engine=VGP",
                kInputStride, kInputBytes);
        return true;
    }

    bool Process(const AX_VIDEO_FRAME_INFO_T& decoded_frame) {
        if (physical_address_ == 0) {
            ++statistics_.errors;
            LogError("[IVPS] Process called before the output buffer was allocated");
            return false;
        }
        if (decoded_frame.stVFrame.u32Width != kSourceWidth ||
            decoded_frame.stVFrame.u32Height != kSourceHeight ||
            decoded_frame.stVFrame.enImgFormat != AX_FORMAT_YUV420_SEMIPLANAR) {
            ++statistics_.errors;
            LogError("[IVPS] expected NV12 %dx%d, actual=%ux%u format=%d",
                     kSourceWidth, kSourceHeight, decoded_frame.stVFrame.u32Width,
                     decoded_frame.stVFrame.u32Height,
                     static_cast<int>(decoded_frame.stVFrame.enImgFormat));
            return false;
        }

        const auto begin = Clock::now();
        if (const auto ret = axclrtMemset(
                reinterpret_cast<void*>(static_cast<std::uintptr_t>(physical_address_)), 0, kInputBytes);
            ret != AXCL_SUCC) {
            ++statistics_.errors;
            LogError("[IVPS] axclrtMemset letterbox buffer failed: 0x%08X",
                     static_cast<unsigned int>(ret));
            return false;
        }

        AX_IVPS_ASPECT_RATIO_T aspect{};
        aspect.eMode = AX_IVPS_ASPECT_RATIO_AUTO;
        aspect.eAligns[0] = AX_IVPS_ASPECT_RATIO_HORIZONTAL_CENTER;
        aspect.eAligns[1] = AX_IVPS_ASPECT_RATIO_VERTICAL_CENTER;
        aspect.nBgColor = 0;

        AX_VIDEO_FRAME_T source = decoded_frame.stVFrame;
        const auto ret = AXCL_IVPS_CropResizeVgp(&source, &output_frame_, &aspect);
        if (ret != AX_SUCCESS) {
            ++statistics_.errors;
            LogError("[IVPS] AXCL_IVPS_CropResizeVgp failed: 0x%08X",
                     static_cast<unsigned int>(ret));
            return false;
        }

        ++statistics_.frames;
        statistics_.total_ms += ElapsedMilliseconds(begin);
        return true;
    }

    bool DumpRawBgr(const std::string& path) const {
        if (path.empty() || physical_address_ == 0) {
            return true;
        }
        std::vector<std::uint8_t> host(kInputBytes);
        const auto ret = axclrtMemcpy(host.data(),
                                      reinterpret_cast<void*>(static_cast<std::uintptr_t>(physical_address_)),
                                      host.size(), AXCL_MEMCPY_DEVICE_TO_HOST);
        if (ret != AXCL_SUCC) {
            LogError("[IVPS] diagnostic D2H readback failed: 0x%08X",
                     static_cast<unsigned int>(ret));
            return false;
        }
        std::ofstream output(fs::path(path), std::ios::binary);
        if (!output) {
            LogError("[IVPS] cannot open dump file: %s", path.c_str());
            return false;
        }
        output.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size()));
        if (!output) {
            LogError("[IVPS] writing dump file failed: %s", path.c_str());
            return false;
        }
        LogInfo("[IVPS] diagnostic raw BGR frame written: %s (%zu bytes)", path.c_str(), host.size());
        return true;
    }

    bool Close() {
        bool successful = true;
        if (physical_address_ != 0 && virtual_address_ != nullptr) {
            successful = RecordCleanupResult("AXCL_SYS_MemFree",
                                             AXCL_SYS_MemFree(physical_address_, virtual_address_));
        }
        physical_address_ = 0;
        virtual_address_ = nullptr;
        output_frame_ = {};
        return successful;
    }

    AX_U64 physical_address() const { return physical_address_; }
    std::size_t bytes() const { return kInputBytes; }
    const IvpsStatistics& statistics() const { return statistics_; }

private:
    AX_U64 physical_address_{0};
    AX_VOID* virtual_address_{nullptr};
    AX_VIDEO_FRAME_T output_frame_{};
    IvpsStatistics statistics_{};
};

struct InferenceStatistics {
    std::uint64_t frames{0};
    std::uint64_t errors{0};
    double inference_total_ms{0.0};
    double postprocess_total_ms{0.0};
};

bool PostprocessYolo26(const ax_runner_tensor_t* outputs, int output_count,
                       int source_width, int source_height,
                       std::vector<detection::Object>* objects) {
    if (outputs == nullptr || objects == nullptr || output_count < 6) {
        LogError("[YOLO26] expected at least 6 output tensors, actual=%d", output_count);
        return false;
    }

    std::vector<detection::Object> proposals;
    float* box_outputs[3] = {
        static_cast<float*>(outputs[0].pVirAddr),
        static_cast<float*>(outputs[2].pVirAddr),
        static_cast<float*>(outputs[4].pVirAddr)};
    float* class_outputs[3] = {
        static_cast<float*>(outputs[1].pVirAddr),
        static_cast<float*>(outputs[3].pVirAddr),
        static_cast<float*>(outputs[5].pVirAddr)};

    for (int index = 0; index < 3; ++index) {
        if (box_outputs[index] == nullptr || class_outputs[index] == nullptr) {
            LogError("[YOLO26] output tensor %d has no host mirror", index);
            return false;
        }
        const int stride = (1 << index) * 8;
        detection::generate_proposals_yolo26(stride, box_outputs[index], class_outputs[index],
                                             kProbabilityThreshold, proposals,
                                             kInputWidth, kInputHeight, kClassCount);
    }

    detection::get_out_bbox(proposals, *objects, kNmsThreshold,
                            kInputHeight, kInputWidth, source_height, source_width);
    return true;
}

class Yolo26Inference final {
public:
    ~Yolo26Inference() {
        Close();
    }

    bool Open(const std::string& model, AX_U64 input_physical_address, std::size_t input_bytes) {
        Close();
        if (runner_.init(model.c_str()) != 0) {
            LogError("[YOLO26] runner.init failed: %s", model.c_str());
            return false;
        }
        opened_ = true;

        if (runner_.get_num_inputs() != 1 || runner_.get_num_outputs() < 6) {
            LogError("[YOLO26] expected 1 input and >=6 outputs, actual=%d/%d",
                     runner_.get_num_inputs(), runner_.get_num_outputs());
            Close();
            return false;
        }
        const auto& input = runner_.get_input(kModelGroupId, 0);
        const std::vector<unsigned int> expected_shape{1, kInputHeight, kInputWidth, kInputChannels};
        if (input.vShape != expected_shape || input.nSize <= 0 ||
            static_cast<std::size_t>(input.nSize) != input_bytes) {
            std::string actual_shape;
            for (const auto dimension : input.vShape) {
                if (!actual_shape.empty()) {
                    actual_shape += 'x';
                }
                actual_shape += std::to_string(dimension);
            }
            LogError("[YOLO26] model must be [1,640,640,3] U8/NHWC/BGR, bytes=%zu; "
                     "actual bytes=%d shape=%s",
                     input_bytes, input.nSize, actual_shape.c_str());
            Close();
            return false;
        }

        const int bind_ret = runner_.set_input(kModelGroupId, 0, input_physical_address,
                                               static_cast<unsigned long>(input_bytes));
        if (bind_ret != 0) {
            LogError("[YOLO26] direct IVPS CMM -> runner.set_input binding failed: 0x%08X",
                     static_cast<unsigned int>(bind_ret));
            Close();
            return false;
        }
        runner_.set_auto_sync_before_inference(false);
        runner_.set_auto_sync_after_inference(true);
        LogInfo("[YOLO26] direct device input bound: phy=0x%llx bytes=%zu "
                "auto_sync_before=false auto_sync_after=true",
                static_cast<unsigned long long>(input_physical_address), input_bytes);
        return true;
    }

    bool Run(std::uint64_t frame_index, int source_width, int source_height) {
        if (!opened_) {
            ++statistics_.errors;
            LogError("[YOLO26] inference requested before model was opened");
            return false;
        }

        if (!warmed_up_) {
            LogInfo("[YOLO26] warmup %d times using the IVPS device buffer", kWarmupCount);
            for (int index = 0; index < kWarmupCount; ++index) {
                if (const int ret = runner_.inference(kModelGroupId); ret != 0) {
                    ++statistics_.errors;
                    LogError("[YOLO26] warmup inference failed: 0x%08X",
                             static_cast<unsigned int>(ret));
                    return false;
                }
            }
            warmed_up_ = true;
        }

        const auto inference_begin = Clock::now();
        if (const int ret = runner_.inference(kModelGroupId); ret != 0) {
            ++statistics_.errors;
            LogError("[YOLO26] inference failed: 0x%08X", static_cast<unsigned int>(ret));
            return false;
        }
        const double inference_ms = ElapsedMilliseconds(inference_begin);

        const auto postprocess_begin = Clock::now();
        std::vector<detection::Object> objects;
        if (!PostprocessYolo26(runner_.get_outputs_ptr(kModelGroupId), runner_.get_num_outputs(),
                               source_width, source_height, &objects)) {
            ++statistics_.errors;
            return false;
        }
        const double postprocess_ms = ElapsedMilliseconds(postprocess_begin);

        ++statistics_.frames;
        statistics_.inference_total_ms += inference_ms;
        statistics_.postprocess_total_ms += postprocess_ms;

        LogInfo("[DETECTION] frame=%llu objects=%zu inference_ms=%.3f",
                static_cast<unsigned long long>(frame_index), objects.size(), inference_ms);
        return true;
    }

    void Close() {
        if (opened_) {
            runner_.release();
            opened_ = false;
        }
        warmed_up_ = false;
    }

    const InferenceStatistics& statistics() const { return statistics_; }

private:
    ax_runner_axcl runner_;
    bool opened_{false};
    bool warmed_up_{false};
    InferenceStatistics statistics_{};
};

void PrintStatistics(const Options& options, Clock::time_point started,
                     const FfmpegRtspDemuxer& demuxer, NativeVdec& vdec,
                     const NativeIvpsPreprocessor* ivps, const Yolo26Inference* yolo,
                     const char* tag) {
    (void)vdec.RefreshStatus();
    const double elapsed = std::max(ElapsedSeconds(started), 0.001);
    const auto& decode = vdec.statistics();
    const double decoded_fps = static_cast<double>(decode.decoded_frames) / elapsed;
    const double send_average_ms = decode.send_calls == 0
                                       ? 0.0
                                       : decode.send_total_ms / static_cast<double>(decode.send_calls);
    const std::uint64_t ivps_frames = ivps == nullptr ? 0 : ivps->statistics().frames;
    const std::uint64_t ivps_errors = ivps == nullptr ? 0 : ivps->statistics().errors;
    const std::uint64_t inference_frames = yolo == nullptr ? 0 : yolo->statistics().frames;
    const std::uint64_t inference_errors = yolo == nullptr ? 0 : yolo->statistics().errors;
    const double ivps_average_ms = ivps_frames == 0
                                       ? 0.0
                                       : ivps->statistics().total_ms / static_cast<double>(ivps_frames);
    const double inference_average_ms = inference_frames == 0
                                            ? 0.0
                                            : yolo->statistics().inference_total_ms /
                                                  static_cast<double>(inference_frames);
    const double postprocess_average_ms = inference_frames == 0
                                              ? 0.0
                                              : yolo->statistics().postprocess_total_ms /
                                                    static_cast<double>(inference_frames);

    LogInfoFlush("[%s] mode=%s elapsed=%.2fs input_packets=%llu attempted_au=%llu sent_au=%llu "
                 "decoded_frames=%llu decoded_fps=%.2f frame=%ux%u format=%d pts_us=%llu "
                 "vdec_errors=%llu send_calls=%llu send_failures=%llu send_task_timeouts=%llu "
                 "slow_sends=%llu full_retries=%llu send_avg_ms=%.3f send_max_ms=%.3f "
                 "vdec_hw_errors=%llu pending_au=%u pending_frames=%u "
                 "ivps_frames=%llu ivps_errors=%llu ivps_avg_ms=%.3f "
                 "infer_frames=%llu infer_errors=%llu infer_avg_ms=%.3f post_avg_ms=%.3f "
                 "ffmpeg_errors=%llu skipped_before_idr=%llu",
                 tag, ModeName(options.mode), elapsed,
                 static_cast<unsigned long long>(demuxer.input_packets()),
                 static_cast<unsigned long long>(decode.attempted_access_units),
                 static_cast<unsigned long long>(decode.sent_access_units),
                 static_cast<unsigned long long>(decode.decoded_frames), decoded_fps,
                 decode.last_width, decode.last_height, static_cast<int>(decode.last_format),
                 static_cast<unsigned long long>(decode.last_pts_us),
                 static_cast<unsigned long long>(decode.errors),
                 static_cast<unsigned long long>(decode.send_calls),
                 static_cast<unsigned long long>(decode.send_failures),
                 static_cast<unsigned long long>(decode.send_runtime_timeouts),
                 static_cast<unsigned long long>(decode.slow_send_calls),
                 static_cast<unsigned long long>(decode.send_full_retries),
                 send_average_ms, decode.send_max_ms,
                 static_cast<unsigned long long>(decode.hardware_decode_errors),
                 decode.left_stream_frames, decode.left_output_frames,
                 static_cast<unsigned long long>(ivps_frames),
                 static_cast<unsigned long long>(ivps_errors), ivps_average_ms,
                 static_cast<unsigned long long>(inference_frames),
                 static_cast<unsigned long long>(inference_errors),
                 inference_average_ms, postprocess_average_ms,
                 static_cast<unsigned long long>(demuxer.ffmpeg_errors()),
                 static_cast<unsigned long long>(demuxer.skipped_before_idr()));
}

enum class ParseOptionsResult {
    kRun,
    kHelp,
    kError,
};

ParseOptionsResult ParseOptions(int argc, char* argv[], Options* options) {
    if (options == nullptr) {
        std::fprintf(stderr, "Internal error: options pointer is null.\n");
        std::fflush(stderr);
        return ParseOptionsResult::kError;
    }
    cmdline::parser command;
    command.add("help", '?', "print this message");
    command.add("no-pause", 0, "do not pause an independently opened Windows console on exit");
    command.add<std::string>("mode", 'r', "vdec-smoke | ivps-smoke | infer", false, "infer");
    command.add<std::string>("source", 's', "RTSP H.264 source URL", false, yolo26_defaults::kRtspSource);
    command.add<std::string>("model", 'm', "YOLO26 AX model path", false, yolo26_defaults::kModelPath);
    command.add<std::string>("config", 'c', "AXCL JSON config path", false, "");
    command.add<std::string>("dump-ivps", 'p', "one-shot 640x640 BGR raw dump path", false, "");
    command.add<int>("device", 'd', "AXCL device-list index", false, 0);
    command.add<int>("duration", 't', "run duration in seconds; 0 means until interrupted", false, 0);
    command.add<int>("read-timeout", 'o', "RTSP open/read timeout in milliseconds", false, 5000);
    command.add<int>("stats-interval", 'i', "statistics interval in seconds", false, 5);
    const bool parsed = command.parse(argc, argv);
    if (command.exist("help")) {
        std::fputs(command.usage().c_str(), stdout);
        std::fflush(stdout);
        return ParseOptionsResult::kHelp;
    }
    if (!parsed) {
        std::fputs(command.error_full().c_str(), stderr);
        std::fputs(command.usage().c_str(), stderr);
        std::fflush(stderr);
        return ParseOptionsResult::kError;
    }

    const std::string mode = command.get<std::string>("mode");
    if (!ParseMode(mode, &options->mode)) {
        std::fprintf(stderr, "Invalid --mode: %s\n", mode.c_str());
        std::fflush(stderr);
        return ParseOptionsResult::kError;
    }
    options->source = command.get<std::string>("source");
    options->model = command.get<std::string>("model");
    options->axcl_config = command.get<std::string>("config");
    options->dump_ivps = command.get<std::string>("dump-ivps");
    options->device_index = command.get<int>("device");
    options->duration_seconds = command.get<int>("duration");
    options->read_timeout_ms = command.get<int>("read-timeout");
    options->statistics_interval_seconds = command.get<int>("stats-interval");

    if (options->source.rfind("rtsp://", 0) != 0 && options->source.rfind("rtsps://", 0) != 0) {
        std::fprintf(stderr, "--source must be an RTSP URL\n");
        std::fflush(stderr);
        return ParseOptionsResult::kError;
    }
    if (options->mode == RunMode::kInfer && !utilities::file_exist(options->model)) {
        std::fprintf(stderr, "Model file does not exist: %s\n", options->model.c_str());
        std::fflush(stderr);
        return ParseOptionsResult::kError;
    }
    if (!options->axcl_config.empty() && !utilities::file_exist(options->axcl_config)) {
        std::fprintf(stderr, "AXCL config file does not exist: %s\n", options->axcl_config.c_str());
        std::fflush(stderr);
        return ParseOptionsResult::kError;
    }
    if (options->device_index < 0 || options->duration_seconds < 0 || options->read_timeout_ms <= 0 ||
        options->statistics_interval_seconds <= 0) {
        std::fprintf(stderr, "Invalid numeric option\n");
        std::fflush(stderr);
        return ParseOptionsResult::kError;
    }
    if (options->mode == RunMode::kVdecSmoke && !options->dump_ivps.empty()) {
        std::fprintf(stderr, "--dump-ivps is unavailable in vdec-smoke mode\n");
        std::fflush(stderr);
        return ParseOptionsResult::kError;
    }
    return ParseOptionsResult::kRun;
}

int Run(const Options& options) {
    InterruptState interrupt{};
    g_interrupt_state = &interrupt;
    bool control_handler_registered = true;
#ifdef _WIN32
    control_handler_registered = SetConsoleCtrlHandler(ConsoleControlHandler, TRUE) != 0;
    if (!control_handler_registered) {
        LogWarning("[SYSTEM] SetConsoleCtrlHandler registration failed");
    }
#else
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);
#endif
    struct ControlHandlerGuard {
        bool registered{false};
        bool active{true};

        bool Release() {
            if (!active) {
                return true;
            }
            active = false;
            bool successful = true;
#ifdef _WIN32
            if (registered && !SetConsoleCtrlHandler(ConsoleControlHandler, FALSE)) {
                LogError("[CLEANUP] SetConsoleCtrlHandler removal failed");
                successful = false;
            }
#else
            std::signal(SIGINT, SIG_DFL);
            std::signal(SIGTERM, SIG_DFL);
#endif
            g_interrupt_state = nullptr;
            return successful;
        }

        ~ControlHandlerGuard() {
            (void)Release();
        }
    } control_handler_guard;
    control_handler_guard.registered = control_handler_registered;

    if (avformat_network_init() < 0) {
        LogError("[FFMPEG] avformat_network_init failed");
        return -1;
    }
    struct NetworkGuard {
        bool active{true};

        bool Release() {
            if (!active) {
                return true;
            }
            active = false;
            const int result = avformat_network_deinit();
            if (result < 0) {
                LogError("[CLEANUP] avformat_network_deinit failed: %s (%d)",
                         AvErrorText(result).c_str(), result);
                return false;
            }
            return true;
        }

        ~NetworkGuard() {
            (void)Release();
        }
    } network_guard;

    FfmpegRtspDemuxer demuxer(&interrupt);
    if (!demuxer.Open(options.source, options.read_timeout_ms)) {
        g_interrupt_state = nullptr;
        return -1;
    }

    AxclEnvironment environment;
    if (!environment.Initialize(options) || !environment.EnsureCurrentContext()) {
        g_interrupt_state = nullptr;
        return -1;
    }

    NativeVdec vdec(&interrupt.stop);
    if (!vdec.Open(kSourceWidth, kSourceHeight)) {
        g_interrupt_state = nullptr;
        return -1;
    }

    NativeIvpsPreprocessor ivps;
    if (options.mode != RunMode::kVdecSmoke && !ivps.Open()) {
        g_interrupt_state = nullptr;
        return -1;
    }

    Yolo26Inference yolo;
    if (options.mode == RunMode::kInfer &&
        !yolo.Open(options.model, ivps.physical_address(), ivps.bytes())) {
        g_interrupt_state = nullptr;
        return -1;
    }

    LogInfo("[CONFIG] mode=%s", ModeName(options.mode));
    LogInfo("[CONFIG] RTSP source=%s", RedactRtspUrl(options.source).c_str());
    LogInfo("[CONFIG] model=%s",
            options.mode == RunMode::kInfer ? options.model.c_str() : "<not loaded>");
    LogInfo("[CONFIG] duration=%s",
            options.duration_seconds == 0 ? "until Ctrl+C" :
                                            std::to_string(options.duration_seconds).c_str());
    LogInfo("[CONFIG] host pixel path=disabled (no decode/resize/CSC)");
    LogInfo("[CONFIG] pipeline=single stream/thread, no reconnect or load-shedding drops; "
            "pre-IDR/corrupt packets may be skipped");

    bool raw_dump_written = false;
    bool failed = false;
    const auto started = Clock::now();
    auto next_statistics = started + std::chrono::seconds(options.statistics_interval_seconds);

    const NativeVdec::FrameHandler frame_handler = [&](const AX_VIDEO_FRAME_INFO_T& frame) {
        if (!environment.EnsureCurrentContext()) {
            return false;
        }
        if (frame.stVFrame.u32Width != kSourceWidth || frame.stVFrame.u32Height != kSourceHeight ||
            frame.stVFrame.enImgFormat != AX_FORMAT_YUV420_SEMIPLANAR) {
            LogError("[VDEC] unexpected frame: %ux%u format=%d pts=%llu",
                     frame.stVFrame.u32Width, frame.stVFrame.u32Height,
                     static_cast<int>(frame.stVFrame.enImgFormat),
                     static_cast<unsigned long long>(frame.stVFrame.u64PTS));
            return false;
        }

        if (options.mode == RunMode::kVdecSmoke) {
            return true;
        }
        if (!ivps.Process(frame)) {
            return false;
        }
        if (!raw_dump_written && !options.dump_ivps.empty()) {
            if (!ivps.DumpRawBgr(options.dump_ivps)) {
                return false;
            }
            raw_dump_written = true;
        }
        if (options.mode == RunMode::kInfer &&
            !yolo.Run(vdec.statistics().decoded_frames,
                      frame.stVFrame.u32Width, frame.stVFrame.u32Height)) {
            return false;
        }
        return true;
    };

    while (!interrupt.stop.load(std::memory_order_relaxed)) {
        const double elapsed = ElapsedSeconds(started);
        if (options.duration_seconds > 0 && elapsed >= options.duration_seconds) {
            interrupt.stop.store(true, std::memory_order_relaxed);
            break;
        }

        std::int64_t read_timeout_us = static_cast<std::int64_t>(options.read_timeout_ms) * 1000;
        if (options.duration_seconds > 0) {
            const auto remaining_us = static_cast<std::int64_t>(
                (static_cast<double>(options.duration_seconds) - elapsed) * 1000000.0);
            read_timeout_us = std::max<std::int64_t>(1, std::min(read_timeout_us, remaining_us));
        }

        AccessUnit access_unit;
        const ReadResult read_result = demuxer.Read(&access_unit, read_timeout_us);
        if (read_result == ReadResult::kInterrupted) {
            if (options.duration_seconds > 0 && ElapsedSeconds(started) >= options.duration_seconds) {
                interrupt.stop.store(true, std::memory_order_relaxed);
                break;
            }
            if (interrupt.stop.load(std::memory_order_relaxed)) {
                break;
            }
            LogError("[FFMPEG] RTSP read interrupted by timeout");
            failed = true;
            break;
        }
        if (read_result == ReadResult::kEof) {
            LogError("[FFMPEG] RTSP stream ended; reconnect is intentionally disabled");
            failed = true;
            break;
        }
        if (read_result == ReadResult::kError) {
            failed = true;
            break;
        }

        if (!environment.EnsureCurrentContext()) {
            failed = true;
            break;
        }
        if (!vdec.Send(access_unit, frame_handler)) {
            if (!interrupt.stop.load(std::memory_order_relaxed)) {
                failed = true;
            }
            break;
        }

        if (Clock::now() >= next_statistics) {
            PrintStatistics(options, started, demuxer, vdec,
                            options.mode == RunMode::kVdecSmoke ? nullptr : &ivps,
                            options.mode == RunMode::kInfer ? &yolo : nullptr, "STATS");
            next_statistics = Clock::now() + std::chrono::seconds(options.statistics_interval_seconds);
        }
    }

    if (!environment.EnsureCurrentContext() || !vdec.Finish(frame_handler)) {
        failed = true;
    }
    PrintStatistics(options, started, demuxer, vdec,
                    options.mode == RunMode::kVdecSmoke ? nullptr : &ivps,
                    options.mode == RunMode::kInfer ? &yolo : nullptr, "FINAL");

    const auto& decode_statistics = vdec.statistics();
    if (decode_statistics.decoded_frames == 0 || decode_statistics.errors != 0 ||
        decode_statistics.hardware_decode_errors != 0 || demuxer.ffmpeg_errors() != 0) {
        LogError("[FINAL] decode validation failed: decoded_frames=%llu vdec_errors=%llu "
                 "vdec_hw_errors=%llu ffmpeg_errors=%llu",
                 static_cast<unsigned long long>(decode_statistics.decoded_frames),
                 static_cast<unsigned long long>(decode_statistics.errors),
                 static_cast<unsigned long long>(decode_statistics.hardware_decode_errors),
                 static_cast<unsigned long long>(demuxer.ffmpeg_errors()));
        failed = true;
    }
    if (options.mode != RunMode::kVdecSmoke &&
        (ivps.statistics().frames == 0 || ivps.statistics().errors != 0)) {
        LogError("[FINAL] IVPS validation failed: frames=%llu errors=%llu",
                 static_cast<unsigned long long>(ivps.statistics().frames),
                 static_cast<unsigned long long>(ivps.statistics().errors));
        failed = true;
    }
    if (options.mode == RunMode::kInfer &&
        (yolo.statistics().frames == 0 || yolo.statistics().errors != 0)) {
        LogError("[FINAL] inference validation failed: frames=%llu errors=%llu",
                 static_cast<unsigned long long>(yolo.statistics().frames),
                 static_cast<unsigned long long>(yolo.statistics().errors));
        failed = true;
    }

    yolo.Close();
    if (!ivps.Close()) {
        failed = true;
    }
    if (!vdec.Close()) {
        failed = true;
    }
    if (!environment.Shutdown()) {
        failed = true;
    }
    demuxer.Close();
    if (!network_guard.Release()) {
        failed = true;
    }
    if (!control_handler_guard.Release()) {
        failed = true;
    }
    FlushApplicationLog();

    return failed ? -1 : 0;
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    ConfigureExitPause(argc, argv);

    Options options;
    const ParseOptionsResult parse_result = ParseOptions(argc, argv, &options);
    if (parse_result == ParseOptionsResult::kHelp) {
        return 0;
    }
    if (parse_result == ParseOptionsResult::kError) {
        return -1;
    }
    if (!InitializeApplicationLogging()) {
        return -1;
    }

    const int result = Run(options);
    LogInfoFlush("[SYSTEM] process exiting with code=%d", result);
    return result;
}

[English](./README_EN.md) | 简体中文

# axcl-samples

## 项目简介

`axcl-samples` 提供在爱芯元智 PCIe 算力卡上运行常见视觉模型的 C++ 示例，基于 AXCL Runtime（运行时）完成设备管理、模型加载和推理。

当前仓库支持 Ubuntu、Debian 和 Windows 11，主要板卡如下：

| 板卡 | 芯片 | 资料 |
| --- | --- | --- |
| AI Core AX-M1 | AX650N | [文档](https://docs.radxa.com/en/aicore/ax-m1) |
| M4Chat | AX8850 | [Wiki](https://wiki.sipeed.com/hardware/zh/maixIV/m4chat/intro.html) |
| LLM-8850 Card | AX8850 | [文档](https://docs.m5stack.com/zh_CN/ai_hardware/LLM-8850_Card) |

本文以 Windows 11、AX8850 和 `axcl_yolo26` 为主要示例。

## 获取代码

```cmd
git clone https://github.com/AXERA-TECH/axcl-samples.git
cd axcl-samples
```

## Windows 11 编译与运行

### 1. 准备环境

需要准备：

- Windows 11 x64；
- Visual Studio 2022，并安装“使用 C++ 的桌面开发”；
- CMake 和 Ninja（构建工具）；
- AXCL Windows x64 SDK（软件开发工具包）、驱动和 Runtime；
- OpenCV Windows x64；
- 与 AX8850 匹配的 `.axmodel` 模型。

以下目录是本文使用的示例，请按实际安装位置调整：

```text
AXCL_DIR   = D:\AXCL\axcl\out\axcl_win_x64
OpenCV_DIR = D:\opencv\opencv\build\x64\vc16\lib
```

配置前确认：

- `AXCL_DIR` 指向的目录下存在 `include`、`lib`、`bin`；
- `lib\libaxcl_rt.lib` 存在；
- `OpenCV_DIR` 指向的目录下存在 `OpenCVConfig.cmake`。

### 2. 检查工具和设备

打开 **Visual Studio 2022 Developer Command Prompt**（开发者命令提示符），不要使用未加载 MSVC（微软 C/C++ 编译器）环境的普通终端。

```cmd
cl
cmake --version
ninja --version
D:\AXCL\axcl\out\axcl_win_x64\bin\axcl-smi.exe
```

如果找不到 Ninja，先把 `ninja.exe` 所在目录加入 `PATH`，例如：

```cmd
set "PATH=D:\ninja-win;%PATH%"
```

`axcl-smi` 应能正常识别 AX8850；否则需要先检查驱动、Runtime 和硬件连接。

### 3. 配置并编译

进入源码目录：

```cmd
cd /d D:\axcl-samples
```

生成 Release 配置：

```cmd
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DAXCL_DIR=D:\AXCL\axcl\out\axcl_win_x64 -DOpenCV_DIR=D:\opencv\opencv\build\x64\vc16\lib
```

只编译 YOLO26 示例：

```cmd
cmake --build build-win --target axcl_yolo26 -j 4
```

生成的程序位于：

```text
build-win\examples\axcl\axcl_yolo26.exe
```

如果切换了编译器、SDK 或 OpenCV 路径，请先删除旧的 `build-win` 目录，再重新配置。

### 4. 运行 YOLO26

程序默认读取 `D:\yolo26\yolo26m.axmodel`，通过 OpenCV 打开源码中配置的 RTSP 实时流，并使用窗口显示识别结果。默认地址可以通过 `--source` 覆盖。

运行前把 AXCL 和 OpenCV 的 DLL（动态链接库）目录临时加入 `PATH`：

```cmd
set "PATH=D:\AXCL\axcl\out\axcl_win_x64\bin;D:\opencv\opencv\build\x64\vc16\bin;%PATH%"
```

使用默认路径直接运行：

```cmd
build-win\examples\axcl\axcl_yolo26.exe
```

也可以通过 `-m` 和 `-s` 临时覆盖模型文件和 RTSP 地址：

```cmd
build-win\examples\axcl\axcl_yolo26.exe -m D:\models\yolo26n.axmodel -s "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

程序优先使用 OpenCV 的 FFmpeg（多媒体编解码库）后端，失败时回退到 OpenCV 自动选择的后端，并在启动时打印实际后端。使用 OpenCV 4.5.2 及以上版本时，会向后端请求将 RTSP 打开和单次读取超时设为 5 秒；旧版本会在启动日志中提示不支持该超时参数。后台取流线程只保留最新帧，识别速度低于码流帧率时会主动丢弃旧帧，避免显示画面持续落后。

预览窗口最大为 `1280×720`，显示检测框、类别、置信度以及以下数据：

- `FPS`：最近一秒实际完成识别并显示的帧数；
- `MAX`：根据最近一秒完整消费端耗时估算的最大处理帧数。

程序启动时会在当前工作目录的 `log` 文件夹中创建 `axcl_yolo26_console_日期_时间.log`，并将原有控制台的标准输出和标准错误统一写入该文件；控制台只保留一行日志路径提示。日志每秒记录取流/解码、预处理、H2D（主机到设备）、NPU（神经网络处理器）执行、D2H（设备到主机）、后处理及绘制显示的耗时，并输出启发式瓶颈判断。`VideoCapture::read()` 的耗时同时包含网络等待、RTSP 处理和 CPU 解码，不能单独视为 CPU 解码耗时。按 `Q`、`Esc` 或关闭窗口退出，退出时在日志中记录全程汇总。

程序不保存结果图片或视频，也不创建 `output` 目录。模型加载、AXCL 初始化及 5 次预热不计入正式统计。启动日志会隐藏 RTSP 密码，但默认地址仍以明文存在于源码中。

## Windows 原生 RTSP 硬解码 + YOLO26

`ax_yolo26_rtsp_native.exe` 是独立目标，不改变上面的 `axcl_yolo26`。它提供三个运行模式：

- `vdec-smoke`：FFmpeg `libavformat` RTSP 解封装 → AXCL Native VDEC；
- `ivps-smoke`：VDEC NV12 → AXCL Native IVPS 640×640 BGR，黑色居中 letterbox；
- `infer`：四路 IVPS 分别写入自己的 CMM（连续媒体内存）最新帧槽，单推理线程将选中帧 D2D
  （设备到设备）复制到 `ax_runner_axcl` 固定输入，再执行 YOLO26 和 CPU 后处理。

该目标默认不参与构建，避免没有 FFmpeg 开发包时影响已有示例。以下命令均在 Visual Studio 2022 Developer Command Prompt（开发者命令提示符）中执行。

进入源码目录并配置运行时 `PATH`：

```cmd
cd /d D:\axcl-samples
set "PATH=D:\ninja-win;D:\AXCL\axcl\out\axcl_win_x64\bin;D:\AXCL\axcl\3rdparty\ffmpeg\win64\lib;D:\opencv\opencv\build\x64\vc16\bin;%PATH%"
```

配置并只编译原生 RTSP 目标：

```cmd
cmake -S . -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DAXCL_DIR=D:\AXCL\axcl\out\axcl_win_x64 -DFFMPEG_DIR=D:\AXCL\axcl\3rdparty\ffmpeg\win64 -DOpenCV_DIR=D:\opencv\opencv\build\x64\vc16\lib -DAXCL_BUILD_YOLO26_RTSP_NATIVE=ON
cmake --build build-native --target ax_yolo26_rtsp_native -j 4
```

`OpenCV_DIR` 必须指向包含 `OpenCVConfig.cmake` 的目录；如果 OpenCV 安装位置不同，请相应替换以上绝对路径。

生成程序位于：

```text
build-native\examples\axcl\ax_yolo26_rtsp_native.exe
```

原生目标与上面的 OpenCV 目标共享默认模型路径和 RTSP 地址。无参数启动时默认进入 `infer` 模式，
并使用同一个地址建立四条独立 RTSP 连接；RTSP 服务端必须允许同一地址同时连接四次：

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe
```

也可以通过 `--model`/`-m` 和 `--source`/`-s` 覆盖默认值。`--source` 仍只接收一个地址，程序会将
它复制给 `camera=0～3`。下列命令中的 RTSP URL 只作为覆盖格式示例，请替换为实际地址。程序日志会
隐藏密码，但共享默认地址仍以明文存在于 `examples\axcl\yolo26_defaults.hpp` 中。

程序通过命令行正常启动后，会在当前工作目录的 `log` 文件夹中创建独立日志文件，命名格式为
`ax_yolo26_rtsp_native_日期_时间_毫秒_pid进程号.log`。应用日志统一带本地毫秒时间戳；逐帧
`[DETECTION]`、周期统计、FFmpeg、AXCL、VDEC、IVPS 和推理日志均写入该文件。AXCL SDK 自身生成的
`axcl_logs.txt` 继续单独保存，不与应用日志合并。

逐帧检测、正常信息和警告不输出到控制台；默认每秒将一条 `[STATS]` 四路增量统计同时写入日志和
原始控制台。统计包含每路 `decoded_fps`、`infer_fps`、`rate_skips`、`busy_drops`，以及合计推理 FPS、
模型平均耗时和累计错误数。应用检测到无法继续运行的错误时，会立即将错误同时写入日志和原始控制台，
并在第一次错误后显示日志绝对路径。`--help` 和参数错误直接显示在控制台，不创建运行日志。日志目录或
文件创建失败时，程序会在控制台报错并停止运行。可用 `--stats-interval` 修改统计间隔，默认值为 `1` 秒。

在 Windows 中直接双击 `.exe`，程序结束后会提示按任意键关闭窗口；从已有 Developer Command Prompt
或其他共享控制台启动时不会暂停。IDE（集成开发环境）或脚本如果为程序创建独立控制台，也可能触发
暂停，自动化场景应显式传入 `--no-pause`。如果缺少 DLL，Windows 加载器可能在程序进入 `main()` 前
终止进程，此时程序内部无法保持控制台窗口，应从 Developer Command Prompt 启动以查看系统错误。
Linux 不启用退出暂停。

日志文件中的每路 `[STATS_DETAIL]` 额外包含 `attempted_au`、`send_calls`、`send_failures`、
`send_task_timeouts`、`recovered_task_timeouts`、`unrecovered_task_timeouts`、
`consecutive_task_timeouts`、`max_consecutive_task_timeouts`、`slow_sends`、`send_avg_ms`、
`send_max_ms`、`latest_replacements` 等累计指标；其中
`send_calls` 还包含队列满重试和 EOS（码流结束标记）发送。单次 `AXCL_VDEC_SendStream` 达到 `50 ms`
会记录慢调用；送流失败时会记录错误码分解、PTS、数据大小和一次故障现场
`AXCL_VDEC_QueryStatus` 快照。Runtime Task（运行时任务）超时时不会重发结果不确定的 AU：设备状态确认
已经接收时继续取帧，未确认接收时丢弃后续非 IDR 帧并从下一个 IDR 恢复；状态异常或连续三次任务超时
视为不可恢复错误，协调停止全部四路并返回非零退出码。

### 阶段一：VDEC smoke

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --mode vdec-smoke --duration 60 --source "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

验收条件：

- 连续运行不少于 60 秒并正常退出；
- 日志出现 `camera=0～3` 四路，四路 `input_packets`、`sent_au`、`decoded_frames` 均持续增加；
- 四个 VDEC Group 均输出 `2560x1440`、NV12，各路 `decoded_fps` 接近视频源帧率；
- 每个 Group 使用 8 个输出帧缓冲，四路合计 32 个，避免沿用原单路 32 个后直接放大四倍 CMM；
- 最终每路日志中 `vdec_errors=0`、`vdec_hw_errors=0`、`ffmpeg_errors=0`；
- `unrecovered_task_timeouts=0`；`send_task_timeouts=0` 最佳，若非零则必须全部计入
  `recovered_task_timeouts`，且 `max_consecutive_task_timeouts < 3`；
- `full_retries` 可以非零，但不能持续增长并导致 FPS 停滞。

### 阶段二：IVPS smoke

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --mode ivps-smoke --duration 60 --dump-ivps native_640x640.bgr --source "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

验收条件：

- 阶段一的条件继续成立；
- 每路 `ivps_frames` 以不超过 10 FPS 的速度增加且 `ivps_errors=0`；
- `native_640x640.bgr` 恰好为 `1228800` 字节；
- 诊断帧是 640×640 packed BGR，2560×1440 内容应缩放为 640×360，并在上、下各产生 140 像素黑边。

`--dump-ivps` 仅回读 `camera=0` 的第一张候选帧用于诊断，不执行 Host resize 或 CSC（色彩空间转换）；
不指定该参数时，IVPS 像素不会回到 Host。

### 阶段三：完整推理

模型默认沿用 `D:\yolo26\yolo26m.axmodel`，也可以用 `--model` 覆盖：

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --mode infer --duration 60 --model "D:\yolo26\yolo26m.axmodel" --source "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

验收条件：

- 启动日志包含 `fixed runner input ready`、`auto_sync_before=false` 和 `auto_sync_after=true`；
- 每路 `infer_fps` 不超过 10，四路公平调度；如果单推理实例不足 40 FPS，旧候选帧会被最新帧覆盖而不积压；
- 四路 `infer_frames` 均持续增加、`infer_errors=0`，日志中的每条 `[DETECTION]` 都包含 `camera=0～3`；
- VDEC、IVPS、FFmpeg 错误计数仍为 0；
- 正式链路没有 Host 视频解码、resize、CSC 或 NPU 输入 H2D（主机到设备）复制；每个候选帧只执行
  一次约 1.2 MB 的设备内 D2D 复制。

首版固定为四路、H.264、2560×1440、RTSP over TCP。四条连接各自使用一个解码线程、Runtime Context
（运行时上下文）、VDEC Group 和 IVPS 最新帧槽；四路共享一个模型和一个推理线程。每路使用单调时钟
限制为最多 10 FPS，四路相位错开 25 ms，不补做历史帧。程序不自动重连；任意一路读取超时、流结束
或处理失败时会记录错误并停止全部路线。`--duration 0` 表示持续运行直到 Ctrl+C 或发生错误，与累计
识别帧数无关。

### 每秒刷新 AX8850 设备状态

`axcl-smi` 本身执行一次后退出。保持推理程序运行，在另一个 PowerShell 窗口进入仓库根目录并执行：

```powershell
.\examples\axcl\monitor_ax_yolo26_rtsp_native.ps1
```

脚本每秒清屏并顺序执行一次 PATH 中的 `axcl-smi.exe`，显示 CPU/NPU（神经网络处理器）利用率、内存、
CMM 和温度等设备级指标；命令返回非零退出码时会显示警告，按 Ctrl+C 停止。如果 `axcl-smi.exe` 不在
PATH，可将绝对路径作为第一个参数：

```powershell
.\examples\axcl\monitor_ax_yolo26_rtsp_native.ps1 "D:\AXCL\axcl\out\axcl_win_x64\bin\axcl-smi.exe"
```

设备级指标与主程序控制台中的四路业务 FPS 分开显示。Linux 主机可直接使用
`watch -n 1 axcl-smi` 达到相同刷新效果。

## Linux 编译简版

确保 AXCL 头文件、运行库和 OpenCV 已安装。默认情况下，项目会从 `/usr/include/axcl` 和 `/usr/lib/axcl` 查找 AXCL。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target axcl_yolo26 -j 4
```

运行示例：

```bash
./build/examples/axcl/axcl_yolo26 -m yolo26n.axmodel -s "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| CMake 找不到 C/C++ 编译器 | 使用 Visual Studio 2022 Developer Command Prompt，并确认 `cl` 可用 |
| 找不到 `OpenCVConfig.cmake` | 将 `OpenCV_DIR` 指向实际包含该文件的目录 |
| 链接时报 `LNK1181` 或找不到 AXCL 库 | 确认 `AXCL_DIR\lib\libaxcl_rt.lib` 存在 |
| 运行时提示缺少 DLL | 将 AXCL 和 OpenCV 的 `bin` 目录加入 `PATH` |
| RTSP 无法打开或没有 FFmpeg 后端 | 确认 OpenCV 启用了 `videoio`/FFmpeg，并确认 OpenCV `bin` 目录中的视频 I/O 与 FFmpeg DLL 可被程序加载 |
| 实际 FPS 明显低于 `MAX` | 先检查摄像头源帧率及控制台中的取流/解码 FPS；`MAX` 不包含等待摄像头送来下一帧的时间 |
| 程序无法发现 AX8850 | 先运行 `axcl-smi`，检查驱动、Runtime 和 PCIe 连接 |
| 修改环境后仍使用旧配置 | 删除 `build-win` 后重新执行 CMake 配置 |

遇到大量编译错误时，优先处理日志中的第一个 `error` 或 `fatal error`，后续错误通常是连锁结果。

## 相关资源

- [AXCL 在线文档](https://axcl-docs.readthedocs.io/zh-cn/latest/)
- [AXCL Windows 环境配置](https://axcl-docs.readthedocs.io/zh-cn/latest/doc_guide_win_setup.html)
- [AXCL-SMI 使用说明](https://axcl-docs.readthedocs.io/zh-cn/latest/doc_guide_axcl_smi.html)
- [YOLO26 模型](https://huggingface.co/AXERA-TECH/yolo26)
- [视觉模型集合](https://huggingface.co/collections/AXERA-TECH/vision-models-67b0bce92ddc61229e8e94ed)
- [ModelScope](https://modelscope.cn/organization/AXERA-TECH)

问题反馈可通过 GitHub Issues；技术交流群：QQ 139953715。

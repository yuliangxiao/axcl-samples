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

## Windows 原生 RTSP/本地文件硬解码 + YOLO26

`ax_yolo26_rtsp_native.exe` 是独立目标，不改变上面的 `axcl_yolo26`。它提供三个运行模式：

- `vdec-smoke`：FFmpeg `libavformat` RTSP、MP4/MOV 或 MPEG-PS 解封装 → AXCL Native VDEC；
- `ivps-smoke`：VDEC NV12 → AXCL Native IVPS 640×640 BGR，黑色居中 letterbox；
- `infer`：四路 IVPS 分别写入自己的 CMM（连续媒体内存）最新帧槽，单推理线程将选中帧 D2D
  （设备到设备）复制到 `ax_runner_axcl` 固定输入，再执行 YOLO26 和 CPU 后处理。

三种模式共用同一套运行期单路 RTSP/VDEC 有界恢复和最终退出判定；初次启动时任一路打开失败仍不重试，
直接完成清理并返回 `-1`。本地文件的正常 EOF（文件结束）不属于故障：程序立即重开文件、保留现有 VDEC
Group，并从下一个 IDR（即时解码刷新）帧开始下一轮。

该目标默认不参与构建，避免没有 FFmpeg 开发包时影响已有示例。以下命令均在 Visual Studio 2022 Developer Command Prompt（开发者命令提示符）中执行。

进入源码目录并配置运行时 `PATH`：

```cmd
cd /d D:\axcl-samples
set "PATH=D:\ninja-win;D:\AXCL\axcl\out\axcl_win_x64\bin;D:\AXCL\axcl\3rdparty\ffmpeg\win64\lib;D:\opencv\opencv\build\x64\vc16\bin;%PATH%"
```

配置并只编译原生 RTSP/本地文件目标：

```cmd
cmake -S . -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DAXCL_DIR=D:\AXCL\axcl\out\axcl_win_x64 -DFFMPEG_DIR=D:\AXCL\axcl\3rdparty\ffmpeg\win64 -DOpenCV_DIR=D:\opencv\opencv\build\x64\vc16\lib -DAXCL_BUILD_YOLO26_RTSP_NATIVE=ON
cmake --build build-native --target ax_yolo26_rtsp_native -j 4
```

`OpenCV_DIR` 必须指向包含 `OpenCVConfig.cmake` 的目录；如果 OpenCV 安装位置不同，请相应替换以上绝对路径。

生成程序位于：

```text
build-native\examples\axcl\ax_yolo26_rtsp_native.exe
```

原生目标沿用默认模型路径。无参数启动时默认进入 `infer` 模式，读取固定的 H.264、2560×1440 文件
`D:\test.mp4`，并将它复制给四条独立解码与识别管线。该文件支持 MP4/MOV 和 MPEG-PS 封装；程序按
内容识别容器，不依赖 `.mp4` 后缀。文件按 DTS（解码时间戳）原速读取，PTS
（显示时间戳）跨循环保持递增，EOF 后持续从头循环；Linux 构建的对应默认路径为 `test.mp4`：

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe
```

`--model`/`-m` 可以覆盖模型路径。显式传入 `--source`/`-s` 时切换到 RTSP，参数必须是
`rtsp://` 或 `rtsps://` URL；程序仍将唯一地址复制给 `camera=0～3`，RTSP 服务端必须允许同一地址
同时连接四次：

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --source "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

日志会隐藏 URL 中的密码。未显式传入 `--source` 时不会使用
`examples\axcl\yolo26_defaults.hpp` 中的默认 RTSP 地址。

程序通过命令行正常启动后，会在当前工作目录的 `log` 文件夹中创建独立日志文件，命名格式为
`ax_yolo26_rtsp_native_日期_时间_毫秒_pid进程号.log`。应用日志统一带本地毫秒时间戳；逐帧
`[DETECTION]`、周期统计、FFmpeg、AXCL、VDEC、IVPS 和推理日志均写入该文件。AXCL SDK 自身生成的
`axcl_logs.txt` 继续单独保存，不与应用日志合并。

日志初始化成功后，控制台先显示一次日志绝对路径。逐帧检测、模型信息和其他正常细节只写日志文件；
默认每秒在控制台输出一条简短摘要，例如
`[STATS] c0 dec=25.0 infer=11.0 | c1 ...`；其中 `dec` 是解码 FPS，`infer` 是推理 FPS。
日志文件中的 `[STATS]` 继续保留每路状态、`rate_skips`、`busy_drops`、合计推理 FPS、模型平均耗时和
明确致命错误数，退出时控制台额外显示一条四路累计帧数 `[FINAL]`。启动日志会声明固定恢复策略：
每路最多 3 次尝试、后续重试间隔 30 秒、稳定窗口 30 秒，运行期可恢复 RTSP 传输故障与已确认的 VDEC
码流故障共用该路预算。警告和错误会立即同时写入日志与控制台；
`--help` 和参数错误直接显示在控制台，不创建运行日志。日志目录或文件创建失败时，程序会在控制台报错
并停止运行。可用 `--stats-interval` 修改统计间隔，默认值为 `1` 秒。

`infer` 模式默认不保存检测图片，也不会创建图片目录、执行全分辨率截图或启动 JPEG 写入线程。显式传入
`--save-images` 后才启用原有保存流程；Windows 默认目录为 `D:\Images`，Linux 默认为当前目录下的
`Images`，可用 `--image-dir` 覆盖，目录不存在时自动创建。启用后，每路健康管线约每秒选择一张与实际
推理严格配对的 2560×1440 原图，绘制检测框、类别、置信度、路号、本地时间和目标数后，以质量 90 的
JPEG 保存。文件名包含路号、本地日期时间、毫秒、PID（进程号）和进程内唯一帧号；若仍发生碰撞则追加
序号，并使用排他创建避免覆盖旧文件。四路每小时会产生约 14400 个文件，程序不会自动删除历史图片。

启用保存后，为保证框与画面属于同一帧，需要截图的 latest-frame（最新帧）槽在共享推理线程领取前
不会被后续候选覆盖；只有该帧额外通过 IVPS 生成全分辨率 BGR 并执行 D2H（设备到主机）回读，四路
每秒各一张时原始回读量约为 42 MiB/s。JPEG 编码和落盘由容量 8 的独立有界队列处理；队列积压、单张
编码或写入失败会记录警告或错误并丢弃对应图片，但不会中断识别。输出目录无法创建时属于启动错误。

完整推理模式会在推理启动满 10 秒后监测每路滚动 10 秒推理 FPS。低于 `10` 时输出性能警告但继续
运行：进入低帧率状态时立即提示，持续异常最多每 30 秒重复一次，恢复后提示一次。单路处于
`reconnecting` 或 `resynchronizing` 时暂停该路监测，并在路线恢复 `running` 后重新累计 10 秒窗口。
单调递增的 `resync_generation` 还能识别两个统计周期之间已开始并完成的短暂重同步，防止旧窗口跨越
包级重同步继续累计。

在 Windows 中直接双击 `.exe`，程序结束后会提示按任意键关闭窗口；从已有 Developer Command Prompt
或其他共享控制台启动时不会暂停。IDE（集成开发环境）或脚本如果为程序创建独立控制台，也可能触发
暂停，自动化场景应显式传入 `--no-pause`。如果缺少 DLL，Windows 加载器可能在程序进入 `main()` 前
终止进程，此时程序内部无法保持控制台窗口，应从 Developer Command Prompt 启动以查看系统错误。
Linux 不启用退出暂停。

日志文件中的每路 `[STATS_DETAIL]` 额外包含 `attempted_au`、`send_calls`、`send_failures`、
`send_task_timeouts`、`recovered_task_timeouts`、`unrecovered_task_timeouts`、
`consecutive_task_timeouts`、`max_consecutive_task_timeouts`、`slow_sends`、`send_avg_ms`、
`send_max_ms`、`latest_replacements`、`vdec_stream_errors`、`last_error_code`、
`corrupt_packets`、`invalid_h264_packets`、`resync_generation`、`recoverable_rtsp_errors`、
`fatal_ffmpeg_errors`、`pending_recovery_errors`、`recovery_attempts`、`recovery_successes`、
`recovery_failures`、`recovered_ffmpeg_errors`、`recovery_downtime_ms`、`file_loops`、
`snapshot_frames` 和 `snapshot_errors` 等累计指标；其中 `ffmpeg_errors` 是
FFmpeg 累计诊断数，不直接决定退出码，
`recovery_attempts/successes/failures` 也是进程生命周期累计值，不充当或重置当前三次预算。
`invalid_h264_packets` 专门统计本地 H.264 结构校验发现并丢弃的包；`resync_generation` 供性能监测丢弃
跨重同步的旧窗口。
`send_calls` 还包含队列满重试和 EOS（码流结束标记）发送。单次 `AXCL_VDEC_SendStream` 达到 `50 ms`
会记录慢调用；送流失败时会记录错误码分解、PTS、数据大小和一次故障现场
`AXCL_VDEC_QueryStatus` 快照。Runtime Task（运行时任务）超时时不会重发结果不确定的 AU：设备状态确认
已经接收时继续取帧，未确认接收时丢弃后续非 IDR 帧并从下一个 IDR 恢复。设备状态异常如果未归类为
已确认暂态错误，仍视为不可恢复错误；连续三次任务超时也会协调停止全部四路并返回 `-1`。

程序成功启动后，RTSP 读取超时、EOF（流结束）、连接重置及已分类网络/I/O（输入输出）错误只触发
故障相机路恢复；`AV_PKT_FLAG_CORRUPT` 标记的单个损坏包计入 `corrupt_packets` 和 `ffmpeg_errors`，
丢弃后等待下一个
IDR（即时解码刷新）关键帧，不立即重连。每次可恢复 RTSP 事件立即计入 `recoverable_rtsp_errors` 和待确认数；
首个健康画面出现时，待确认数转入 `recovered_ffmpeg_errors`。已恢复的历史错误和损坏包不影响健康退出。
本地文件 EOF 是正常循环边界，不增加 FFmpeg 错误、恢复尝试或待确认错误；本地文件打不开、容器不是
MP4/MOV 或 MPEG-PS、编码不是 H.264、分辨率不是 2560×1440，或循环重开失败时直接协调停止四路。

> **运行期 H.264 包级重同步**：输入会话成功打开后（包括首个 IDR 到达前），运行期包未设置
> `AV_PKT_FLAG_CORRUPT`，但本地校验发现空包、非 Annex-B、无有效
> NAL（网络抽象层单元）或 NAL 头、类型、载荷长度无效时，程序增加
> `invalid_h264_packets`、丢弃该包并开始包级重同步。未处于会话恢复的路线发布 `resynchronizing`；仍在
> 恢复且尚无健康解码帧的路线按优先级继续发布 `reconnecting`。后续非 IDR 画面继续丢弃，首个有效 IDR
> 只有在 VDEC 送流成功后才结束包级重同步；非恢复路线恢复 `running`，恢复路线仍须等首帧和 VDEC 状态
> 健康后才能恢复 `running`。读取截止前仍无有效 IDR 时才进入上述单路恢复。空包检查位于可选 BSF（比特流
> 过滤器）之前，其他 Annex-B/NAL 检查针对可选 BSF 输出；初始输入规格、BSF、内存、内部状态和资源
> 错误仍立即全局失败。

完整边界见[单路有界恢复需求](docs/requirements/2026-08-24-172111-rtsp-vdec-单路有界恢复.md)和
[ADR 0002](docs/adr/0002-use-idr-resynchronization-for-runtime-malformed-packets.md)。

VDEC 仅将 SDK 明确返回的 `AX_ERR_VDEC_STRM_ERROR` 作为单路可恢复码流错误；没有目标 SDK 明确语义
依据时不扩展其他暂态白名单，因此 `format_err`、`pic_size_err`、`stream_unsupported`、`pack_err`、
`ref_err` 等状态诊断不单独推导为可恢复。Context（运行时上下文）绑定、状态查询、普通 VDEC API、
旧资源或部分新资源清理、IVPS 以及共享推理错误仍为全局致命错误。诊断默认不保存原始 H.264 视频数据。

每路独立拥有三次恢复机会，RTSP 与上述 VDEC 故障共用：第 1 次在旧会话完整关闭后立即重建；第 1、2
次失败后各等待 30 秒；第 3 次失败立即发布 `failed`，协调停止全部路线并返回 `-1`，不存在第 4 次。
重建 RTSP/VDEC 连接本身不算成功；只有产生首个正常解码画面且 VDEC 状态正常后才进入 `running`，
并开始 30 秒稳定窗口。窗口内再次故障仍使原尝试失败并继续消耗原预算；连续健康 30 秒后恢复完整三次
预算。恢复触发、尝试开始/失败、首个健康帧、预算重置和耗尽日志均包含 `camera`、分类、尝试序号与
剩余预算。多路同时故障时预算互不借用，其他健康路及共享推理继续工作；任一路先耗尽即全局关停。

### 三模式共同恢复与退出验收

下列条件分别适用于 `vdec-smoke`、`ivps-smoke` 和 `infer`。应使用能定向中断单个 RTSP 客户端会话的
受控服务端或代理，并根据日志时间戳和分类字段验收；整机断网不能证明健康相机路仍在继续处理。
带 `resynchronizing` 或 `invalid_h264_packets` 的条目需要受控坏包注入和目标机日志证明；仅有源码静态检查
不得标记为运行验收通过。

- 健康基线下四路持续处理并返回 `0`；初次任一路打不开时没有恢复尝试并返回 `-1`。
- 运行期单路超时、EOF、Windows 原始 Winsock 连接重置值 `-10054` 或 FFmpeg 已归一化网络/I/O 错误
  只使该路进入 `reconnecting`；第 1 次无等待，健康路帧数继续增加，`infer` 的共享推理继续运行。
- 持续不可达时日志只有 attempt 1、2、3；attempt 2、3 分别在前次失败约 30 秒后开始，第 3 次失败后
  没有 attempt 4，全部路线完成清理并返回 `-1`。
- RTSP/VDEC 交错故障及首帧后 30 秒内复发继续使用原预算；连续健康 30 秒后的下一次故障从 attempt 1
  重新开始。多路同时故障各自计数，任一路先耗尽即全局失败。
- `AV_PKT_FLAG_CORRUPT` 包增加 `corrupt_packets`；运行期本地校验发现的可丢弃码流包增加
  `invalid_h264_packets`。两者均不送入 VDEC，而是开始包级重同步并等待有效 IDR；未处于会话恢复时对外
  进入 `resynchronizing`。IDR 成功送流后不消耗恢复预算且最终健康结束返回 `0`；超时后才按当前已用
  预算开始下一次单路恢复，当前没有已用预算时为 attempt 1。
- 非 H.264、分辨率变化、无效 extradata、BSF、SPS/PPS 缓存超限、内存/内部状态、Context、VDEC
  清理、IVPS 或共享推理故障不消耗恢复预算并立即全局失败。
- Ctrl+C 或 `--duration` 到期时，首个健康帧已经出现但尚在稳定窗口内的 `running` 路线可以返回 `0`；
  任一路仍为 `resynchronizing`、`reconnecting` 或 `failed` 时完成有序清理并返回 `-1`。

### 阶段一：VDEC smoke

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --mode vdec-smoke --duration 60 --source "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

验收条件：

- 连续运行不少于 60 秒并正常退出；
- 日志出现 `camera=0～3` 四路，四路 `input_packets`、`sent_au`、`decoded_frames` 均持续增加；
- 四个 VDEC Group 均输出 `2560x1440`、NV12，各路 `decoded_fps` 接近视频源帧率；
- 每个 Group 使用 8 个输出帧缓冲，四路合计 32 个，避免沿用原单路 32 个后直接放大四倍 CMM；
- 最终每路日志中 `vdec_errors=0`、`vdec_current_hw_errors=0`、`fatal_ffmpeg_errors=0`，且退出时不处于
  `resynchronizing`/`reconnecting`/`failed`；历史 `vdec_stream_errors`、`recoverable_rtsp_errors`、
  `corrupt_packets`、`invalid_h264_packets` 和 `ffmpeg_errors` 可以非零，已恢复的 RTSP 错误应计入
  `recovered_ffmpeg_errors`；
- `send_task_timeouts=0` 最佳；若非零，应全部计入 `recovered_task_timeouts`，且
  `max_consecutive_task_timeouts < 3`；
- `full_retries` 可以非零，但不能持续增长并导致 FPS 停滞。

### 阶段二：IVPS smoke

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --mode ivps-smoke --duration 60 --dump-ivps native_640x640.bgr --source "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

验收条件：

- 阶段一的条件继续成立；
- 每路 `ivps_frames` 长期以不超过 11 FPS 的速度增加且 `ivps_errors=0`；
- `native_640x640.bgr` 恰好为 `1228800` 字节；
- 诊断帧是 640×640 packed BGR，2560×1440 内容应缩放为 640×360，并在上、下各产生 140 像素黑边。

`--dump-ivps` 仅回读 `camera=0` 的第一张候选帧用于诊断，不执行 Host resize 或 CSC（色彩空间转换）；
不指定该参数时，IVPS 像素不会回到 Host。

### 阶段三：完整推理

模型默认沿用 `D:\yolo26\yolo26m.axmodel`，也可以用 `--model` 覆盖：

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --mode infer --duration 60 --model "D:\yolo26\yolo26m.axmodel"
```

上例使用默认 `D:\test.mp4`。切换到 RTSP 时显式增加 `--source`：

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --mode infer --duration 60 --model "D:\yolo26\yolo26m.axmodel" --source "rtsp://user:password@192.168.0.201:554/Streaming/Channels/101"
```

以上命令默认不保存图片。如需启用原有图片保存流程，增加 `--save-images`；还可同时用 `--image-dir`
指定保存目录：

```cmd
build-native\examples\axcl\ax_yolo26_rtsp_native.exe --mode infer --duration 60 --model "D:\yolo26\yolo26m.axmodel" --save-images --image-dir "D:\Images"
```

验收条件：

- 启动日志包含 `fixed runner input ready`、`auto_sync_before=false` 和 `auto_sync_after=true`；
- 每路候选上限为 11 FPS，推理启动满 10 秒后的滚动 10 秒 `infer_fps` 应不低于 10，且四路公平调度；
  如果单推理实例不足以处理约 44 FPS，旧候选帧会被最新帧覆盖而不积压；
- 四路 `infer_frames` 均持续增加、`infer_errors=0`，日志中的每条 `[DETECTION]` 都包含 `camera=0～3`；
- 默认启动日志包含 `image_saving=disabled`，且不创建图片目录；传入 `--save-images` 后，`D:\Images`
  中每路约每秒产生一张 2560×1440 JPEG，`snapshot_frames` 持续增加，正常磁盘负载下保存线程的
  `dropped/errors` 应为 0；
- VDEC 致命错误、当前硬件错误、`fatal_ffmpeg_errors` 和 IVPS 错误计数仍为 0；允许历史
  `vdec_stream_errors`、`recoverable_rtsp_errors`、`corrupt_packets`、`invalid_h264_packets` 和
  `ffmpeg_errors` 非零；
- 正式链路没有 Host 视频解码、resize、CSC 或 NPU 输入 H2D（主机到设备）复制；每个候选帧只执行
  一次约 1.2 MB 的设备内 D2D。默认不执行截图 D2H；传入 `--save-images` 后，每路每秒选中的截图帧
  会额外执行一次全分辨率 IVPS CSC 和约 10.55 MiB D2H，其他候选帧不回读原图。

当前实现固定为四路、H.264、2560×1440，支持默认 MP4/MOV 或 MPEG-PS 本地文件循环，以及显式 RTSP over TCP。四条管线各自使用一个解码线程、Runtime Context
（运行时上下文）、VDEC Group 和 IVPS 最新帧槽；四路共享一个模型和一个推理线程。每路使用固定单调
时间轴限制为最多 11 FPS，四路相位按约 90.909 ms 的周期均匀错开；错过的节拍直接跳过，只处理最新帧，
不补做历史帧。运行期已分类的 RTSP 传输故障与 `AX_ERR_VDEC_STRM_ERROR` 按上述共享预算执行单路重建；
当前实现中的其他输入、内部状态和共享处理错误仍停止全部路线，只放宽其中可在送入 VDEC 前隔离的运行期
单包内容错误。`--read-timeout` 只控制单次 RTSP 打开或读取超时，
不改变恢复次数、间隔或稳定窗口；IDR 重同步也复用固定读取截止时间，同一轮不会因状态事件、
再次读取、连续坏包或连续非 IDR 刷新该时间，也不增加独立参数或坏包频率阈值。`--duration 0` 表示持续
运行直到 Ctrl+C 或发生错误，与累计识别帧数无关；到期或用户停止时仍有
路线处于 `resynchronizing`/`reconnecting`/`failed` 会返回 `-1`。

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

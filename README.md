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

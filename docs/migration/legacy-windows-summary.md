# Windows 旧实现迁移摘要

## 迁移决定

- 迁移日期：2026-09-02。
- 新项目以 AXERA-TECH 官方 `axcl-samples/main` 为代码基线。
- 初始化基线提交：`cbfa4c76891758983ca2b0c99c11d6621d59af39`。
- 保留官方上游历史，放弃旧仓库的 19 个自定义提交、未提交修改、运行日志和仓库外教程。
- 当前阶段只完成仓库初始化和历史记录，不迁移或重新实现旧功能。
- 后续开发目标为 Ubuntu 22.04 x86_64 主机和 AX8850 PCIe 算力卡。

官方资料：

- AXCL 示例：https://github.com/AXERA-TECH/axcl-samples
- AXCL 文档：https://axcl-docs.readthedocs.io/zh-cn/latest/
- 模型转换文档：https://github.com/AXERA-TECH/pulsar2-docs

## 旧 Windows 环境

- Windows 11 x64。
- Visual Studio 2022 Community，MSVC 19.44.35228。
- CMake 3.31.12，Ninja 1.13.x。
- OpenCV 4.14.0。
- AXCL Windows x64 SDK 与 Runtime 3.16.0。
- AX8850 PCIe 算力卡。

## 旧实现功能

### 普通 YOLO26

- 通过 OpenCV 读取本地视频或 RTSP。
- OpenCV 优先使用 FFmpeg 后端，失败时回退到自动后端。
- 支持模型路径、视频源和输入尺寸命令行参数。
- 模型预热后执行推理并显示检测结果。
- 使用第一块可用 AXCL 设备。

### 原生四路媒体管线

- 支持 `vdec-smoke`、`ivps-smoke` 和 `infer` 三种模式。
- 四路 H.264 输入，旧测试规格为 2560×1440、25 FPS。
- FFmpeg 使用 RTSP over TCP 解封装。
- AXCL VDEC 硬件解码，IVPS 转换为 640×640 BGR。
- 四路各自维护解码和 IVPS 管线，共享单个 YOLO26 推理线程。
- 每路候选推理上限约 11 FPS。
- 支持无效 H.264 包丢弃、IDR 重同步和单路有界恢复。
- 支持可选 JPEG 截图，旧实现约每路每秒保存一张。
- Windows 下曾使用 PowerShell 脚本每秒执行 `axcl-smi.exe` 监控设备。

### 模型约束

- 原生管线使用 640×640、U8、NHWC、BGR 输入。
- 旧 YOLO26 模型至少包含六个 FP32 输出，按 box/class 交替排列。
- 旧模型按 80 个类别处理。

## 旧实现已知问题

- 长时间运行日志中曾出现 `device 4 is dead`。
- 设备异常后业务统计冻结，但进程继续周期输出日志。
- 日志没有最终统计或正常退出记录。
- RTSP 断流、坏包、IDR 重同步、设备掉线恢复和有序退出尚未完成系统验收。

## Ubuntu 22.04 测试服务器现状

2026-09-02 只读核查结果：

- Ubuntu 22.04.5 LTS，Linux 5.15，x86_64。
- Intel N100，4 核 4 线程。
- PCIe 可枚举 AXERA 设备 `1f4b:0650`。
- Secure Boot 已关闭，未发现 IOMMU 已启用的迹象。
- 根分区约有 47 GB 可用空间。
- 尚未安装 GCC、G++、CMake、Make、Ninja 和 pkg-config。
- 尚未安装 OpenCV 与 FFmpeg。
- 尚未安装或识别 AXCL Host 驱动、Runtime、头文件、库和 `axcl-smi`。
- PCIe 设备当前没有可见的 AXCL 内核驱动绑定，因此尚不能验证卡的固件和健康状态。

服务器和摄像头的临时测试凭据保存在仓库根目录的 `.env.local`。该文件只供本地 Codex 使用，并由 Git 忽略；公开仓库只包含 `.env.example` 占位模板。

## 后续开发前置工作

1. 获取与实际 AX8850 卡匹配的 AXCL Host 驱动、Runtime 和设备固件版本。
2. 安装 Ubuntu 22.04 编译工具、OpenCV 和 FFmpeg 开发包。
3. 使用 `axcl-smi` 验证驱动、设备和固件状态。
4. 记录 AXCL、驱动、固件、模型及官方示例提交的精确版本。
5. 在新代码中按需求重新实现并验收多路 RTSP、VDEC、IVPS、YOLO26 推理、恢复和截图功能。

多路媒体功能开始实现时，可评估 `ax-video-sdk` 和 `ax-pipeline`；初始化阶段不引入这些额外依赖。

# AXERA is pleased to support the open source community by making ax-samples available.
#
# Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# Author:
#

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

include("${CMAKE_SOURCE_DIR}/cmake/axcl.cmake")

axera_example(axcl_classification ax_classification_steps.cc)

axera_example(axcl_yolov5s ax_yolov5s_steps.cc)
axera_example(axcl_yolov5s_seg ax_yolov5s_seg_steps.cc)
axera_example(axcl_yolov5_face ax_yolov5_face_steps.cc)
axera_example(axcl_yolov7_face ax_yolov7_face_steps.cc)
axera_example(axcl_yolov8 ax_yolov8_steps.cc)
axera_example(axcl_yolov8_seg ax_yolov8_seg_steps.cc)
axera_example(axcl_yolov8_pose ax_yolov8_pose_steps.cc)
axera_example(axcl_yolov9 ax_yolov9_steps.cc)
axera_example(axcl_yolov9_u ax_yolov9_ultralytics_steps.cc)
axera_example(axcl_yolov10 ax_yolov10s_steps.cc)
axera_example(axcl_yolov10_u ax_yolov10_ultralytics_steps.cc)
axera_example(axcl_yolo11 ax_yolo11_steps.cc)
axera_example(axcl_yolo11_seg ax_yolo11_seg_steps.cc)
axera_example(axcl_yolo11_pose ax_yolo11_pose_steps.cc)
axera_example(axcl_yolo26 ax_yolo26_steps.cc)
target_link_libraries(axcl_yolo26 PRIVATE Threads::Threads)

option(AXCL_BUILD_YOLO26_RTSP_NATIVE
       "Build the FFmpeg demux + AXCL Native VDEC/IVPS YOLO26 RTSP sample"
       OFF)

if(AXCL_BUILD_YOLO26_RTSP_NATIVE)
    if(NOT AXCL_DIR)
        message(FATAL_ERROR "AXCL_DIR is required to build ax_yolo26_rtsp_native")
    endif()

    if(NOT FFMPEG_DIR AND DEFINED ENV{FFMPEG_DIR})
        file(TO_CMAKE_PATH "$ENV{FFMPEG_DIR}" FFMPEG_DIR)
    endif()
    if(NOT FFMPEG_DIR)
        get_filename_component(_axcl_source_root "${AXCL_DIR}/../.." ABSOLUTE)
        set(_axcl_bundled_ffmpeg "${_axcl_source_root}/3rdparty/ffmpeg/win64")
        if(EXISTS "${_axcl_bundled_ffmpeg}/include/libavformat/avformat.h")
            set(FFMPEG_DIR "${_axcl_bundled_ffmpeg}")
        endif()
    endif()
    if(NOT FFMPEG_DIR)
        message(FATAL_ERROR
                "FFMPEG_DIR is required. It must contain include/libavformat/avformat.h and lib/*.lib")
    endif()

    find_path(AXCL_RTSP_FFMPEG_INCLUDE_DIR
              NAMES libavformat/avformat.h
              PATHS "${FFMPEG_DIR}/include"
              NO_DEFAULT_PATH)
    find_library(AXCL_RTSP_AVFORMAT_LIBRARY
                 NAMES libavformat avformat
                 PATHS "${FFMPEG_DIR}/lib"
                 NO_DEFAULT_PATH)
    find_library(AXCL_RTSP_AVCODEC_LIBRARY
                 NAMES libavcodec avcodec
                 PATHS "${FFMPEG_DIR}/lib"
                 NO_DEFAULT_PATH)
    find_library(AXCL_RTSP_AVUTIL_LIBRARY
                 NAMES libavutil avutil
                 PATHS "${FFMPEG_DIR}/lib"
                 NO_DEFAULT_PATH)

    find_path(AXCL_RTSP_AXCL_INCLUDE_DIR
              NAMES axcl.h axcl_vdec.h axcl_ivps.h
              PATHS "${AXCL_DIR}/include"
              NO_DEFAULT_PATH)
    find_library(AXCL_RTSP_SYS_LIBRARY
                 NAMES libaxcl_sys axcl_sys
                 PATHS "${AXCL_DIR}/lib"
                 NO_DEFAULT_PATH)
    find_library(AXCL_RTSP_VDEC_LIBRARY
                 NAMES libaxcl_vdec axcl_vdec
                 PATHS "${AXCL_DIR}/lib"
                 NO_DEFAULT_PATH)
    find_library(AXCL_RTSP_IVPS_LIBRARY
                 NAMES libaxcl_ivps axcl_ivps
                 PATHS "${AXCL_DIR}/lib"
                 NO_DEFAULT_PATH)

    foreach(_dependency
            AXCL_RTSP_AXCL_INCLUDE_DIR
            AXCL_RTSP_FFMPEG_INCLUDE_DIR
            AXCL_RTSP_AVFORMAT_LIBRARY
            AXCL_RTSP_AVCODEC_LIBRARY
            AXCL_RTSP_AVUTIL_LIBRARY
            AXCL_RTSP_SYS_LIBRARY
            AXCL_RTSP_VDEC_LIBRARY
            AXCL_RTSP_IVPS_LIBRARY)
        if(NOT ${_dependency})
            message(FATAL_ERROR
                    "ax_yolo26_rtsp_native dependency ${_dependency} was not found. "
                    "AXCL_DIR=${AXCL_DIR}; FFMPEG_DIR=${FFMPEG_DIR}")
        endif()
    endforeach()

    axera_example(ax_yolo26_rtsp_native ax_yolo26_rtsp_native.cc)
    target_include_directories(ax_yolo26_rtsp_native PRIVATE
                               "${AXCL_RTSP_AXCL_INCLUDE_DIR}"
                               "${AXCL_RTSP_FFMPEG_INCLUDE_DIR}")
    target_link_libraries(ax_yolo26_rtsp_native PRIVATE
                          Threads::Threads
                          "${AXCL_RTSP_SYS_LIBRARY}"
                          "${AXCL_RTSP_VDEC_LIBRARY}"
                          "${AXCL_RTSP_IVPS_LIBRARY}"
                          "${AXCL_RTSP_AVFORMAT_LIBRARY}"
                          "${AXCL_RTSP_AVCODEC_LIBRARY}"
                          "${AXCL_RTSP_AVUTIL_LIBRARY}")

    message(STATUS "ax_yolo26_rtsp_native AXCL_DIR: ${AXCL_DIR}")
    message(STATUS "ax_yolo26_rtsp_native FFMPEG_DIR: ${FFMPEG_DIR}")
endif()

axera_example(axcl_yolo26_depth ax_yolo26_depth_steps.cc)
axera_example(axcl_yolo_world_open_vocabulary ax_yolo_world_open_vocabulary_steps.cc)
axera_example(axcl_palm_detection ax_palm_detection_steps.cc)
axera_example(axcl_handpose ax_handpose_steps.cc)
axera_example(axcl_depth_anything ax_depth_anything_steps.cc)

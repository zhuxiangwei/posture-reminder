# -*- coding: utf-8 -*-
"""坐姿小卫士 · 运行时包

模块划分（职责单一，改哪类问题去哪个模块）：

    paths      路径与目录（config/baseline/profile/voice/review/logs/models 的唯一来源）
    deps       第三方依赖的延迟导入与缺依赖报错（全项目只有这里 import cv2/mediapipe）
    compat_matplotlib  Windows SAC 拦 matplotlib 时的兼容桩
    texts      给孩子/家长看的文案常量（界面与引擎共用一份，避免两处漂移）
    metrics    坐姿指标与判定：正面/侧面、合理性筛除、基线、阈值建议、标定自检
    gate       迟滞与提醒间隔（ReminderGate）—— CLI 与 GUI 共用同一份实现
    settings   配置读写、基线读写、默认参数、运行时把配置灌进 metrics
    camera     ★ 摄像头子系统：档位阶梯、开流重试、子进程探测、profile 记忆、
               采集线程（CameraStream）、掉线自动重连
    pose       姿态推理封装（landmarker 构造、读帧+推理、画关键点）
    voice      语音播放（winsound / MCI 双通道 + 音量档位 + 播放线程）
    review     判定现场留存与家长审核（误报率的唯一数据来源）
    logutil    运行日志落盘
    engine     后台线程引擎：采集 → 推理 → 判定 → 迟滞 → 提醒，对外只给 snapshot()
    cli        命令行入口（--list-cameras / --bench / --video 回放 / 实时窗口）
    ui         PySide6 图形界面（theme / widgets / main_window / parent_dialog /
               review_panel / selftest）

入口脚本在包外：`posture_app.py`（图形界面）、`posture_monitor.py`（命令行）。
"""

__all__ = ["camera", "cli", "compat_matplotlib", "deps", "engine", "gate", "logutil",
           "metrics", "paths", "pose", "review", "settings", "texts", "voice"]

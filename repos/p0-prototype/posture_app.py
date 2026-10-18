#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐姿小卫士 · 图形界面（PySide6 / Qt）
=====================================

给**孩子**用的界面。目标：双击就能跑，不需要看懂任何数字和参数。

    python posture_app.py            # 正常启动（自动开始监测）
    python posture_app.py --demo     # 演示模式：不连摄像头，循环展示各种状态
    python posture_app.py --selftest # 只构建界面并退出（离屏，冒烟测试）

为什么用 PySide6 而不是 tkinter
-------------------------------
tkinter 依赖 Python 发行版**自带** tcl/tk。实测本机的 Python 3.13 便携版不含 tkinter，
`import tkinter` 直接 ModuleNotFoundError；而 Qt 是自带二进制的 pip 包，
装完就能用，不受发行版影响。另外 QSS 能做圆角大按钮、平滑字体，更适合儿童界面。

设计原则（都来自报告 §8 的儿童场景研究）
----------------------------------------
1. **零操作启动**：双击即开始。孩子不该需要按任何按钮才能被提醒。
2. **先看脸色，再看字**：状态用大色块 + 手绘表情表达，文字只是补充。
3. **不用红色**：警示用暖橙。红 = 惩罚感，会让孩子想关掉程序。
4. **正向激励**：坐得好时是绿色大字「坐得真棒！」，并把"坐得好的时间""坐姿良好率"
   "连续良好"放在最显眼处，而不是只显示犯了几次错。
5. **设置与孩子隔离**：阈值、标定、提醒节奏全部收进「家长设置」，孩子界面看不到。

技术要点
--------
- 界面这边**只有 QTimer 读快照**，绝不跨线程碰控件；推理与判定在 `PostureEngine`
  的独立线程里，摄像头又在 `CameraStream` 的**另一条**线程里（2026-10-18 起）。
  ⚠️ 采集必须是**一条长命线程**：DSHOW 走 COM，设备句柄绑创建它的线程，
  "临时起个线程开流再交给别人读"拿到的是死句柄（实测 0/5）。
  开流进度、降档警告、掉线重连都从 `snapshot()["cam"]` / `cam_warnings` 出来，不许忽略
- 预览帧用 QImage.Format_BGR888 直接包装 numpy 缓冲，**不需要编解码往返**
- 表情用 QPainter 画，不用 emoji —— emoji 字体在不同 Windows 上差异大，可能变方块
- 图片一律走 `ScaledImageLabel`：缩放在 `paintEvent` 里按**当前**控件尺寸 × DPR 算。
  在 `setPixmap` 时固化尺寸会随布局回流变成"只显示一角"（HiDPI 下的老 bug）
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from posture.app import main           # noqa: E402  界面全部代码在 posture/ui/*

if __name__ == "__main__":
    main()

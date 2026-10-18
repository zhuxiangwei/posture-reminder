#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行时引擎（兼容壳）

真正的实现在 `posture/engine.py`。本文件只做 re-export，
让老的 `import posture_engine as pe` 写法（启动器、测试、docs 里的示例）继续可用。

    eng = PostureEngine(cfg)
    eng.start()
    snap = eng.snapshot()            # 任意时刻可读，返回一份浅拷贝
    eng.pause() / resume() / recalibrate() / stop()

★ 引擎的结构变化（这才是"加载很久"和"画面僵住"的根治处）
  重构前：引擎线程**自己**开流、自己 `cap.read()`。于是
      ① 开流的 5~20s 里界面干等、且没有任何进度；
      ② `cap.read()` 在 C 层阻塞时整条循环跟着僵，界面假死；
      ③ 摄像头被抢 / USB 掉线后**没有恢复路径** —— 画面停在最后一帧，
         程序看起来还在跑，实际早就不更新了。
  现在：设备归 `posture/camera.CameraStream`（自己的长命线程，
      **创建与读取同线程**，DSHOW/COM 的硬要求），引擎只"取最新帧 → 推理 → 判定"，
      并把 `opening / ready / reconnecting / failed` 实时通过 `snapshot()["cam"]`
      报给界面。另外两条顺手的收益：
      · 模型构造与开流**并行**（create_from_video 要 0.3~1s 且不碰设备）；
      · 基线按**实际**分辨率校验（降档后像素类判据会静默失效，属于漏报）。

snapshot() 新增的两项（界面必须处理，别忽略）
      cam           dict：phase / backend / camera / width / height /
                          degraded / reconnects / short / stale_sec
      cam_warnings  list[str]：非空就必须显示 —— 里面是"只拿到 1080p，
                    低于请求的 4K"这类**不许静默**的警告
"""

from posture import engine                                     # noqa: F401
from posture.engine import PostureEngine                        # noqa: E402,F401

# 引擎历史上"顺手"提供的周边（界面与测试都当公开面在用，一并转发）
from posture.gate import (CLEAR_SEC, REMIND_GAP_ANY_SEC,        # noqa: E402,F401
                          REMIND_GAP_SEC, SUSTAIN_SEC, ReminderGate)
from posture.logutil import RunLog                              # noqa: E402,F401
from posture.pose import PREVIEW_MAX_W, shrink                  # noqa: E402,F401
from posture.review import ReviewStore                          # noqa: E402,F401
from posture.settings import (DEFAULT_CAM_H, DEFAULT_CAM_W,     # noqa: E402,F401
                              DEFAULT_CONFIG, FPS_SAMPLE,
                              apply_config, baseline_reason, load_baseline,
                              load_config, save_baseline, save_config)
from posture.texts import (ISSUE_TIP, TEXT_GOOD, TEXT_IDLE,     # noqa: E402,F401
                           TEXT_NODETECT, TEXT_PAUSED)
from posture.voice import Speaker                               # noqa: E402,F401

# 旧私有名：界面与测试都直接引用过，保留转发
_shrink = shrink

__all__ = [
    "PostureEngine", "RunLog", "ReviewStore", "Speaker",
    "DEFAULT_CONFIG", "DEFAULT_CAM_W", "DEFAULT_CAM_H", "FPS_SAMPLE",
    "PREVIEW_MAX_W", "SUSTAIN_SEC", "CLEAR_SEC", "REMIND_GAP_SEC",
    "REMIND_GAP_ANY_SEC", "ReminderGate",
    "apply_config", "load_config", "save_config", "load_baseline",
    "save_baseline", "baseline_reason",
    "TEXT_GOOD", "TEXT_NODETECT", "TEXT_PAUSED", "TEXT_IDLE", "ISSUE_TIP",
    "shrink", "_shrink", "engine",
]

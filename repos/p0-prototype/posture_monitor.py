#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口（兼容壳）· 坐姿实时提醒 P0 原型

用法不变：

    python posture_monitor.py --list-cameras
    python posture_monitor.py --side-only --camera 0 --calib 10
    python posture_monitor.py --bench --bench-frames 100
    python posture_monitor.py --video 录像.mp4 --calib 10

⚠️ 本文件现在**只是一层壳**，实现全部在 `posture/` 包里。
   搬家的原因就是要治用户报的两条症状（"加载很久""出错时不知道发生了什么"）：
   原来一个 1742 行的文件里同时住着摄像头开流、指标数学、迟滞、语音、CLI 主循环，
   改摄像头要在一堆判定函数里翻，改判定要怕碰坏开流 —— 结果就是不敢改，
   而"不敢改"在用户那边的表现是同一个错误反复出现。
   现在按职责拆开：

       posture/camera    ★ 开流、档位阶梯、后端探测、profile 记忆、采集线程、掉线重连
       posture/metrics   指标与判定（纯数学，不碰设备）
       posture/gate      迟滞与提醒间隔
       posture/settings  配置与基线（含"基线绑定实际分辨率"的校验）
       posture/pose      MediaPipe 推理封装
       posture/voice     语音
       posture/review    判定现场留存与家长审核
       posture/logutil   运行日志落盘
       posture/cli       命令行逻辑（本文件对应它）
       posture/deps      cv2/mediapipe 的延迟导入与缺依赖报错

为什么还留着这个壳（两条都是硬约束）：
  ① 三个 `.bat` 启动器、README、docs 里写的都是这个文件名；
  ② `tests/test_offline.py` 与外部脚本长期 `import posture_monitor as pm`。
所以这里的 re-export 是**公开的兼容面**，删名字等于破坏性改动。
新代码请一律 `from posture import ...`。

★ 唯一要当心的兼容点：`SIDE_YAW_DEG`
  它是 `posture.metrics` 的**模块级参数**（判定函数直接读它）。
  在这个壳里再存一份副本会变成"假开关"：赋值了但判定不生效 —— 比报错更糟。
  所以本壳用 PEP 562 的 `__getattr__` 做**只读转发**，并给 `set_side_yaw()` 做写入口。
  （Python 的模块属性**赋值**没有钩子，所以 `pm.SIDE_YAW_DEG = 45` 这种老写法
    只会写进壳自己的字典、不影响判定；测试与脚本请改用
    `posture_monitor.set_side_yaw(45)` 或直接写 `posture.metrics.SIDE_YAW_DEG`。）
"""

import sys

# ---- 设备 / 采集 ----------------------------------------------------------
from posture import camera                                         # noqa: F401
from posture.camera import (BACKEND_NAMES, CAM_TIERS, CHECKLIST,   # noqa: E402,F401
                            CameraError, CameraProfile, CameraStream,
                            MAX_OPEN_ATTEMPTS, MJPG_FOURCC,
                            MIN_ATTEMPT_SEC, OPEN_BUDGET_SEC,
                            OPEN_PROBE_TIMEOUT, OpenResult, REOPEN_DELAY_SEC,
                            STALL_SEC, VideoSource,
                            backend_viable_subprocess, frame_is_usable,
                            list_cameras, open_camera, open_camera_result,
                            open_once, read_frame, read_frame_timed,
                            tier_ladder, wait_usable_frame, warmup_budget)

# ---- 依赖与路径 -----------------------------------------------------------
from posture import deps as _deps                                  # noqa: F401
from posture.deps import (MJPG_FOURCC, cv2, mp, mp_available,       # noqa: E402,F401
                          mp_python, need_cv, need_mp, vision)
from posture.paths import (BASELINE_PATH, CAMERA_PROFILE_PATH,      # noqa: E402,F401
                           CONFIG_PATH, DEFAULT_MODEL, LOG_DIR,
                           MODEL_DIRS, PKG_DIR, REVIEW_DIR, ROOT,
                           VOICE_DIR, find_model)

# ---- 指标与判定 -----------------------------------------------------------
from posture import metrics as mx                                  # noqa: F401
from posture.metrics import (ISSUE_METRIC, L_EAR, L_HIP, L_SH,      # noqa: E402,F401
                             NOSE, R_EAR, R_HIP, R_SH, SANE,
                             SIDE_SANE, SIDE_THRESHOLDS, THRESHOLDS,
                             build_baseline, compute_metrics,
                             compute_side_metrics, fuse, judge,
                             judge_side, sanity_rows, side_score,
                             suggest_threshold)

# ---- 迟滞 / 提醒 ----------------------------------------------------------
from posture import gate                                           # noqa: F401
from posture.gate import (CLEAR_SEC, REMIND_GAP_ANY_SEC,            # noqa: E402,F401
                          REMIND_GAP_SEC, SUSTAIN_SEC, GateState, ReminderGate)
from posture.voice import Speaker, play_reminder                   # noqa: E402,F401

# ---- 推理封装 -------------------------------------------------------------
from posture import pose                                            # noqa: F401
from posture.pose import (KEYPOINTS_DRAW, PREVIEW_MAX_W, draw,      # noqa: E402,F401
                          grab, make_landmarker, shrink)

# ---- 配置 / 基线 ----------------------------------------------------------
from posture import settings                                        # noqa: F401
from posture.settings import (DEFAULT_CAM_H, DEFAULT_CAM_W,         # noqa: E402,F401
                              DEFAULT_CONFIG, FPS_SAMPLE, apply_config,
                              baseline_mismatch, baseline_reason, load_baseline,
                              load_config, read_baseline_raw, save_baseline,
                              save_config)

# ---- 文案 ----------------------------------------------------------------
from posture import texts                                           # noqa: F401
from posture.texts import (HEAD_FWD, HEAD_TILT, ISSUES, ISSUE_TIP,  # noqa: E402,F401
                           LOOK_DOWN, ROUNDED_BACK, SHOULDER_SLOPE, SIDE_ISSUES,
                           SLUMP, TAG_FRONT, TAG_SIDE, TEXT_GOOD, TEXT_IDLE,
                           TEXT_NODETECT, TEXT_PAUSED)

# ---- 存储 / 日志 ----------------------------------------------------------
from posture.logutil import RunLog                                  # noqa: E402,F401
from posture.review import ReviewStore                              # noqa: E402,F401

# ---- 命令行逻辑 -----------------------------------------------------------
from posture import cli                                             # noqa: F401
from posture.cli import (bench, build_cameras, build_parser, main,   # noqa: E402,F401
                         print_sanity, replay)


# ---------------------------------------------------------------- 兼容别名

_fmt_metrics = cli._fmt_metrics
_mmss = cli._mmss
_load_baseline_json = cli._load_baseline_json
_need_cv = _deps.need_cv
_need_mp = _deps.need_mp
_shrink = shrink


def set_side_yaw(deg):
    """设置侧向机位偏航角（判定函数的模块级参数）。

    ⚠️ 必须走这里、或直接写 `posture.metrics.SIDE_YAW_DEG` ——
       写在本壳上的 `SIDE_YAW_DEG = x` **不会**影响判定（见模块文档）。
    """
    mx.SIDE_YAW_DEG = float(deg)
    return mx.SIDE_YAW_DEG


def __getattr__(name):
    """PEP 562：把 `SIDE_YAW_DEG` 的读操作转发到真正的归属地。"""
    if name == "SIDE_YAW_DEG":
        return mx.SIDE_YAW_DEG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(cli.run())

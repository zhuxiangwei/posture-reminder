#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐姿实时提醒 · P0 原型
======================

目标：在本机跑通「采集 -> 姿态推理 -> 判定 -> 语音提醒」全链路，并可直接实测性能。

两种模式
--------
1) **纯侧面**（`--side-only`，纸笔作业场景推荐，**本机当前采用**）
   摄像头侧放到孩子侧方 80~90°，只用一颗。本机只接了一颗 USB 摄像头，索引 0。
   测：头前倾、塌腰、低头（头下探）。
   放弃：歪头 / 歪肩（冠状面，侧面原理上测不到）、「含胸驼背」融合升级项。
   详见 docs/03-side-only-plan.md。

2) **双机位**（需两颗摄像头；本机只有一颗，暂不可用）
   正面机位：低头 / 歪头 / 歪肩 / 距离过近。
   侧面机位：头前倾 / 塌腰。
   两路互证时升级判为「含胸驼背」——这是单路做不到的。

三条设计约束（见 docs/01-research-report.md）
  1. 判定用「相对个人基线的偏移」，不用绝对阈值——阈值随体型/机位/镜头变化，绝对值不可靠。
  2. 坐姿变化是秒级慢过程 -> 采样 5 FPS 足够。单帧越界绝不报警，必须连续超标才提醒。
  3. 隐私：全程本地推理，不上传任何画面。

⚠️ 一、角度的宽高比校正（2026-09-22 修复）
  MediaPipe 输出的是**归一化坐标**：x 除以画面宽、y 除以画面高，两者尺度不同。
  直接 atan2(dx_norm, dy_norm) 得到的**不是真实角度**——16:9 画面下角度会被压扁约 44%，
  真实 12° 前倾只显示成约 6.8°，导致侧面阈值实际偏松、漏报。
  必须先乘回像素宽高再算角度。这就是 compute_side_metrics 里每条坐标都 `* w` / `* h` 的原因。

⚠️ 二、侧向机位的偏航角（--side-yaw）
  摄像头越接近正侧（yaw=90°），前后方向的形变越小；45° 斜侧只能看到前后位移的
  sin(45°)≈71%。程序按 1/sin(yaw) 把水平分量还原回矢状面，使 SIDE_THRESHOLDS 始终是真实度数。
  **纯侧面模式建议直接摆 80~90°，不必用 45°。**

用法：
    python posture_monitor.py --side-only --camera 0 --calib 10   # 纯侧面（推荐）
    python posture_monitor.py --camera 0 --camera2 1 --calib 10   # 双机位
    python posture_monitor.py --bench --bench-frames 100          # 性能自测
    python posture_monitor.py --list-cameras                      # 摄像头排查第一步

离线自检（不需要摄像头、不需要 cv2/mediapipe）：
    python tests/test_offline.py
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------- 依赖（延迟报错）
#
# 刻意不在这里 sys.exit：判定逻辑（compute_* / judge_* / fuse）是纯数学，
# 离开 cv2/mediapipe 也应能 import，这样 tests/test_offline.py 可在任何机器上回归。
try:
    import cv2
except ImportError:
    cv2 = None

# mediapipe 的 vision 包会无条件 `import matplotlib.pyplot`（只为那个 3D 调试
# 可视化函数 plot_landmarks）。本机 Windows 智能应用控制（SAC）拦掉了
# matplotlib/_image.cp313-win_amd64.pyd，导致整包导入失败 —— 在这里先用桩顶掉，
# 详见 compat_matplotlib.py 的模块文档。真实 matplotlib 可用时它什么都不做。
try:
    import compat_matplotlib
    compat_matplotlib.install()
except Exception:
    pass

_MP_IMPORT_ERR = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as _e:
    # 刻意把原始报错留下来：SAC 拦截时抛的是
    #   ImportError: DLL load failed while importing _image:
    #   应用程序控制策略已阻止此文件。
    # 以前这里只写 `except ImportError: vision = None`，异常被吞掉，
    # 最后在引擎里变成一个莫名其妙的
    #   AttributeError: 'NoneType' object has no attribute 'PoseLandmarker'
    # 真病因完全看不到。留着原文，_need_mediapipe() 会把它一起报出来。
    _MP_IMPORT_ERR = _e
    mp = mp_python = vision = None


def _need_cv():
    if cv2 is None:
        sys.exit("缺少依赖。请先执行： pip install mediapipe opencv-python")


def _need_mp():
    if mp is None or vision is None:
        msg = "缺少 mediapipe（或 mediapipe 导入失败）。请先执行： pip install mediapipe"
        if _MP_IMPORT_ERR is not None:
            # 把真正的异常原文附上，不要只报一个 None。
            msg += f"\n\n导入 mediapipe 时的原始报错：\n  {type(_MP_IMPORT_ERR).__name__}: {_MP_IMPORT_ERR}"
            if "应用程序控制策略" in str(_MP_IMPORT_ERR) or "Smart App Control" in str(_MP_IMPORT_ERR):
                msg += ("\n\n这看起来是 Windows 智能应用控制（SAC）拦截了第三方扩展 DLL。\n"
                        "  排查：设置 > 隐私和安全性 > Windows 安全中心 > 应用和浏览器控制 > 智能应用控制\n"
                        "  项目已内置兼容桩 compat_matplotlib.py；若仍报此错，说明被拦的不止 matplotlib，\n"
                        "  把上面那段原始报错发出来定位具体文件。")
        sys.exit(msg)


# ---------------------------------------------------------------- 配置

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [
    os.path.join(HERE, "models"),
    os.path.join(HERE, "..", "..", "bench", "models"),
]
DEFAULT_MODEL = "pose_landmarker_lite.task"

# ---------------- 正面视角阈值：相对基线的偏离量（比例） ----------------
#
# ⚠️ 以下为「未标定的初始值」，必须用真实数据回归（见任务书 §6 的 P-3 阶段）。
#    标定方法：录一段孩子正常写作业的视频，标出真实塌腰/歪头的时间点，
#    扫阈值看误报率与漏报率，取等错误率点。切勿直接拿去长期使用。
THRESHOLDS = {
    # 头部（耳-肩垂直距离）相对基线的收缩比例。
    # 0.10 对应头部下移约 3~4cm，是肉眼可辨的塌腰起点。
    "head_drop": 0.10,
    "head_tilt": 0.06,    # 双耳高度差（归一化）
    "shoulder": 0.05,     # 双肩高度差（归一化）
    "too_close": 0.15,    # 头肩尺度相对基线的膨胀比例
}

# ---------------- 侧面视角阈值：相对基线的**增量**（真实度数 / 比例） ----------------
#
# ⚠️ 同样是未标定的初始值。侧面阈值以「度」为单位，已做宽高比与偏航角校正。
SIDE_THRESHOLDS = {
    "neck_angle": 12.0,    # 颈部前倾角相对基线增加这么多「度」算头前倾
    "trunk_angle": 8.0,    # 躯干倾角相对基线增加这么多「度」算塌腰
    "head_collapse": 0.12,  # 肩-耳距离相对基线收缩这么多比例算低头/趴桌
}

# 「这一帧的关键点是否可信」的合理性区间。
# 用途：挡掉检测器的误检帧——画面里没有人时，检测器偶尔会误报出一团挤在一起的关键点，
#       若不筛掉，标定阶段会把垃圾样本当成基线（实测踩过，见 compute_metrics 注释）。
# 这些是宽松的物理常识边界，不是精度阈值，宁松勿严。
SANE = {
    "span_min": 0.06,       # 肩宽至少占画面宽度 6%（否则是误检或人太远）
    "span_max": 0.95,       # 肩宽不可能占满画面
    "head_drop_min": 0.4,   # 耳-肩距至少是肩宽的 0.4 倍（再小说明关键点错乱）
    "head_drop_max": 4.0,   # 至多 4 倍
}

SIDE_SANE = {
    "neck_angle_max": 60.0,   # 侧面颈部前倾角不可能超过 60°
    "trunk_angle_max": 45.0,  # 躯干倾角不可能超过 45°
    "len_frac_min": 0.03,     # 肩-耳距离至少占画面短边 3%（再小说明关键点挤成一团）
}

SUSTAIN_SEC = 4.0     # 连续超标多久才提醒
CLEAR_SEC = 3.0       # 恢复正常多久才解除
REMIND_GAP_SEC = 90.0  # **同类**问题两次提醒的最小间隔（防同一问题反复叫）
# **任意两次提醒之间**的下限。
# ⚠️ 它必须是全局的、不能按问题分别计时 —— 否则孩子扭一下，
#    问题名在 头前倾 / 塌腰 / 低头 之间轮换，每个问题各自都没到间隔，
#    却会轮流触发，变成每几秒响一次（实测踩过）。
# 需求原话：「三十秒之内如果提示过不正确就不提示，即便不正确的原因也不一样。」
REMIND_GAP_ANY_SEC = 30.0
FPS_SAMPLE = 5        # 判定采样帧率（推理频率）

# 侧向机位的偏航角：0°=正对，90°=正侧。纯侧面模式建议 80~90°。
# 只影响侧面指标（用于把被压缩的前后位移还原回矢状面真实量）。
SIDE_YAW_DEG = 90.0

# MediaPipe 33 点索引
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24

# ⚠️ 采集分辨率：默认 **3840x2160（4K）+ MJPG**。2026-09-28 实测定稿。
#
#   当前摄像头：USB 2.0 Camera（VID_0C49&PID_636F，**标称 4K**，走 UVC）。
#
#   ★ 全部谜底在于**压缩格式**（用户点破"这是 4K 摄像头"后才查到的）：
#     OpenCV/DSHOW 默认协商 **YUY2 未压缩**，而 USB 2.0 实际带宽只有 ~35MB/s：
#       480p YUY2  = 0.6MB/帧 -> 20 FPS，100% 可靠
#       720p YUY2  = 1.8MB/帧 -> 10 FPS，**开流成功率仅 60%**（逼近带宽极限）
#       1080p YUY2 = 4.0MB/帧 -> 3.3 FPS，开流成功率 62%（超极限，很不稳）
#     → **"这摄像头不行"的假象全是 YUY2 未压缩造成的**。
#     切到 **MJPG 压缩**后（CAP_PROP_FOURCC=MJPG，且必须设在分辨率之前）：
#       3840x2160 MJPG: **10/10 = 100% 可靠，~14 FPS**（实测一次跑到 19 FPS）
#       2560x1440 MJPG: 10/10 = 100%，~14 FPS
#       1080p：驱动**不提供** MJPG（强设被忽略、退回 YUY2），故 1080p 反而不可取
#     这也解释了「为什么系统相机应用正常」：Media Foundation 会自动协商压缩格式。
#
#   4K 全管线实测（读帧 35.3 + cvtColor 3.0 + 推理 14.2 = **52.6ms/帧**，
#   预算 200ms@5FPS，占 1/4）。另注：MediaPipe 输入固定 256x256，
#   分辨率不影响姿态精度 —— 4K 的收益在「提醒现场存证照片」，不在判定。
#
#   代价要在意一条：`review_keep_fullres=True` 时存证照片是 4K JPEG（每张 ~1-3MB），
#   默认上限 300 条 → 最多约 1GB。磁盘紧张就在界面关掉"存原图"。
#
#   历史留档（2026-09-28 早些时候的错误结论，记下来防重犯）：
#     - 曾以为"1080p 是上限、720p 是甜点" —— 那是 YUY2 下的假象；
#     - 曾写"480p 会把肩耳距离压到卡门槛" —— 错，门槛 0.03*min(w,h) 随分辨率缩放，
#       余量分辨率无关（各档均 2.1~2.5x）。
#   更早（Nebula 02 时代）：只有 MSMF 能用、固定 1080p —— 换摄像头结论完全反转，
#   **换设备必须重测**。
DEFAULT_CAM_W = 3840
DEFAULT_CAM_H = 2160

MJPG_FOURCC = cv2.VideoWriter_fourcc(*'MJPG') if cv2 is not None else 0
"""UVC 压缩格式。⚠️ 必须**在设置分辨率之前** set，格式协商发生在选模式时。
驱动若不提供 MJPG 会静默忽略（退回 YUY2）—— 那是 1080p 的情形，属正常。"""


def find_model(name=DEFAULT_MODEL):
    for d in MODEL_DIRS:
        p = os.path.normpath(os.path.join(d, name))
        if os.path.isfile(p):
            return p
    sys.exit(
        f"找不到模型 {name}。\n"
        f"下载地址（注意档位名在路径中出现两次）：\n"
        f"  https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_lite/float16/1/pose_landmarker_lite.task\n"
        f"放到以下任一目录：\n  " + "\n  ".join(os.path.normpath(d) for d in MODEL_DIRS)
    )


# ⚠️ 后端必须挨个试，不能写死 —— 换一颗摄像头结论就可能完全反转。
# 历史实测（留档，说明"不能写死"这条为什么重要）：
#   2026-09-22 / 09-26（Nebula 02, VID_3AAE&PID_6373）：
#       DSHOW 按索引 0~3 全部 open 失败，只有 MSMF 能打开，固定 1920x1080。
#   2026-09-28（新摄像头 USB 2.0 Camera, VID_0C49&PID_636F）：
#       结论**完全反转** —— DSHOW 能开；MSMF 的 grab 报 E_PENDING，
#       且 **`cv2.VideoCapture()` 构造函数本身就无限阻塞**（全新进程单独
#       构造也复现，不是被前面的开合搞坏的状态）。
# 所以顺序只能作为"优先猜测"，真正决定用哪个必须靠**能否读到有效帧**；
# 会阻塞的后端则必须靠**子进程探测**排除（见 backend_viable_subprocess）。
BACKENDS = []


def _build_backends():
    global BACKENDS
    if not BACKENDS and cv2 is not None:
        BACKENDS = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY),
        ]
    return BACKENDS


def read_frame(cap):
    """安全读帧。返回 (ok, frame)。

    ⚠️ 必须包 try —— 实测（MSMF 后端 + 设置分辨率）：
        cv2.error: OpenCV(5.0.0) ... (-215:Assertion failed) _step >= minstep
        in function 'cv::Mat::Mat'      at cap.read()
    驱动异常时 cap.read() **会直接抛 cv2.error，而不是规规矩矩返回 (False, None)**。
    原实现所有 read 都没保护，摄像头打一次嗝就能让整个程序崩掉。
    """
    try:
        ok, frame = cap.read()
    except Exception:
        return False, None
    if not ok or frame is None:
        return False, None
    return True, frame


def read_frame_timed(cap, timeout=2.5):
    """带**超时保护**的读帧，返回 (ok, frame, timed_out)。

    ⚠️ 为什么普通 try/except 不够：
      实测本机 MSMF 后端 grab 失败后（Error: -2147483638 / E_PENDING），
      cap.read() 会**阻塞数分钟不返回**。这是 C 层阻塞，try/except 抓不住，
      只会让整个程序（含 GUI 线程）永久卡死。
      所以探测阶段必须用工作线程 + join 超时兜住。

    卡住的线程是 daemon，不会阻止进程退出；但它持有的设备句柄要到
    cap.release() 才释放，因此调用方仍需负责 release。
    """
    box = {}

    def _rd():
        try:
            ok, fr = cap.read()
            box["r"] = (ok, fr)
        except Exception:
            box["r"] = (False, None)

    th = threading.Thread(target=_rd, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return False, None, True        # 卡住，视为不可用
    ok, fr = box.get("r", (False, None))
    if not ok or fr is None:
        return False, None, False
    return True, fr, False


def frame_is_usable(frame, min_max_pixel=8):
    """判断一帧是否**真的有画面**（而不只是"读到了"）。

    ⚠️ 这是 2026-09-28 踩出来的坑，务必保留：
       cap.read() 返回 ok=True **不代表画面有效**。
       实测本机 USB 2.0 Camera（VID_0C49&PID_636F）在它不支持的档位下：
         · read() 返回 True
         · frame.shape 正常（如 1080x1920x3）
         · 但**内容全是 0**（frame.max() == 0），且每帧耗时精确 1000ms
       这是驱动"接受了格式协商但填不进数据"的占位行为。
       只判 ok 就会选中这个组合，然后画面全黑、判定逻辑对着黑图跑。

    判据用 max() 而非 mean()：只要有一个像素非零就说明有真实数据流，
    避免把"画面确实很暗"的正常情况误判成故障。
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    try:
        return int(frame.max()) >= min_max_pixel
    except Exception:
        return False


def warmup_budget(width=None, height=None):
    """该分辨率档位允许的"全零预热帧"上限（实测值 + 余量）。

    ⚠️ 为什么要按档位区分（2026-09-28 实测的逐帧序列）：
       好的开流前几帧也是全零的，档位越高越多：
         640x480   : 首帧就出图        -> 0 帧预热
         1280x720  : 第 1 帧出图       -> 1 帧预热
         1920x1080 : 第 3 帧才出图     -> 3 帧预热
       而**坏开流**的帧是每 1000ms 一张全零（掉帧占位）。
       所以这个上限既是"别误杀好开流"的下界，也是"别在坏开流上白等"的上界：
       一刀切设 4（为了兼容 1080p）会让 720p 的每次坏开流白等 4s 而不是 2s，
       实测把启动耗时从 ~4s 拖到 ~10s。故按档位给值。
    """
    h = height or DEFAULT_CAM_H
    return 4 if h > 900 else 2


def wait_usable_frame(cap, tries=5, timeout=2.5, blank_abort=None):
    """连续读若干帧，只要有一帧**有效**就返回它。

    为什么要多试几帧：UVC 摄像头开流最初几帧常是空的（曝光/白平衡未就绪），
    只读一帧容易把好设备误判成坏设备。

    ⚠️ blank_abort 的取值有实测依据，**不能一刀切**（2026-09-28 两次踩坑）：
       本机 USB 2.0 Camera 的"**好的**开流"前几帧也是全零的，档位越高越多：
         640x480 : 首帧就出图（0 帧预热）
         1280x720: 第 1 帧出图（1 帧预热）
         1920x1080: 第 3 帧出图（**3 帧预热**）
       · 设成 2 的坑：1080p 的每个好开流都在第 2 帧被误杀，
         表现为"1080p 重试四次全失败"，其实设备完好。
       · 一刀切设 4 的坑：兼容了 1080p，却让 720p 的每次**坏**开流
         白等 4×1000ms（坏帧就是 1000ms 一张），启动被拖慢一倍以上。
       → 所以默认交给 warmup_budget(档位) 按需取值，别写死。

    另：坏开流的帧是每 1000ms 一张全零（掉帧占位），好开流预热后
    会骤降到 ~50ms(480p) / ~100ms(720p) / ~305ms(1080p)。
    耗时无法用来提前判别（好开流首帧同样要等 1000ms），只能靠帧数。

    返回 (ok, frame, stats)；stats 带诊断信息，供报错时说明失败类型。
    """
    if blank_abort is None:
        blank_abort = warmup_budget()
    last = None
    timed_out = False
    blank_run = 0
    for i in range(tries):
        ok, fr, to = read_frame_timed(cap, timeout)
        if to:
            timed_out = True
            break
        if ok:
            last = fr
            if frame_is_usable(fr):
                return True, fr, {"tries": i + 1, "max_pixel": int(fr.max()),
                                  "timed_out": False}
            blank_run += 1
            if blank_abort and blank_run >= blank_abort:
                break      # 预热帧数已超上限仍全零 -> 这轮开流是坏的
    mp = int(last.max()) if last is not None else -1
    return False, last, {"tries": tries, "max_pixel": mp, "timed_out": timed_out}


MAX_OPEN_ATTEMPTS = 5
"""同一 (后端, 索引, 分辨率) 组合最多重开几次。

⚠️ 为什么需要重开而不是简单重读帧（2026-09-28 实测）：
   本机 USB 2.0 Camera（VID_0C49&PID_636F）**开流有失败率**，但这不是
   "档位不支持" —— 同一档位重开就能好。实测各档成功率（每档 8 轮独立开合）：
     640x480   8/8 = 100%
     1280x720  4~5/8 ≈ 50~62%
     1920x1080 5/8 = 62%
   所以重读同一句柄没用（那一轮开流本身就是坏的），必须**换新句柄重开**。

   取 5 次是权衡：按单次成功率 50%（保守估计）算，
     3 次 -> 87.5%，4 次 -> 93.8%，5 次 -> 96.9%
   再往上加，收益变小而最坏启动耗时线性增长（每次坏尝试约 4s，
   见 wait_usable_frame 的 blank_abort），所以停在 5。
"""

OPEN_PROBE_TIMEOUT = 8.0
"""回退后端探测的超时（秒）。见 backend_viable_subprocess()。

⚠️ 为什么用**子进程**而不是线程超时：
   某些后端的 `cv2.VideoCapture()` **构造函数本身会无限阻塞**（实测 MSMF），
   而这是 C 层阻塞 —— 线程 join 超时救不回来（那个线程会永远卡住并攥着设备）。
   子进程可以被硬杀，是唯一可靠的办法。

⚠️ 为什么**不能**"在工作线程里创建、再交给别的线程用"（2026-09-28 实测的 2x2 对照）：
   每格 5 轮、各自独立进程，1280x720 DSHOW：
       创建线程    读取线程    成功率
       主线程      主线程      4/5 = 80%
       主线程      工作线程    3/5 = 60%
       **工作线程  主线程      0/5 = 0%**
       **工作线程  工作线程    0/5 = 0%**
   DSHOW 走 COM，cap 绑定创建它的线程的 COM 单元；创建线程一结束，cap 就是死的。
   **结论：创建必须发生在真正要用它的那个线程里。**
   （读取线程则无所谓 —— C1/C2 都在 60~80% 区间，说明读取可跨线程。）

所以：**主路径在调用线程直接创建**（不套线程），只有回退到"可疑后端"时才先
用子进程探一下可用性，避免主进程被卡死。
"""

_BACKEND_VIABLE_CACHE = {}


def backend_viable_subprocess(name, index, width, height, timeout=None):
    """在**子进程**里验证 (后端, 索引, 分辨率) 能否读到有效画面。

    返回 True/False。子进程超时（即构造函数或读帧阻塞）一律视为不可用，
    并把它硬杀 —— 这样卡死的后端不会拖住主进程。
    结果按 (name,index,width,height) 缓存，避免重复开销。
    """
    timeout = timeout if timeout is not None else OPEN_PROBE_TIMEOUT
    key = (name, index, width, height)
    if key in _BACKEND_VIABLE_CACHE:
        return _BACKEND_VIABLE_CACHE[key]

    code = (
        "import cv2, sys\n"
        f"cap = cv2.VideoCapture({index}, cv2.CAP_{name})\n"
        "if not cap.isOpened():\n"
        "    print('NO'); sys.exit(0)\n"
        f"cap.set(cv2.CAP_PROP_FRAME_WIDTH, {width})\n"
        f"cap.set(cv2.CAP_PROP_FRAME_HEIGHT, {height})\n"
        "best = 0\n"
        "for _ in range(10):\n"
        "    ok, fr = cap.read()\n"
        "    if ok and fr is not None and fr.size > 0:\n"
        "        best = max(best, int(fr.max()))\n"
        "        if best > 8:\n"
        "            break\n"
        "print('YES' if best > 8 else 'NO')\n"
    )
    ok = False
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        ok = "YES" in (r.stdout or "")
    except subprocess.TimeoutExpired:
        ok = False          # 卡住了 -> 该后端不可用（子进程已被 kill）
    except Exception:
        ok = False
    _BACKEND_VIABLE_CACHE[key] = ok
    return ok


def open_camera(idx=None, backend=None, width=None, height=None):
    """打开摄像头。

    索引、后端、分辨率**都**不可靠，逐个组合尝试：
      - 索引：接多个摄像头时 0 未必是想要的那颗
      - 后端：别人机器上的结论**不能照搬**。实测同一台机器换颗摄像头就反转：
          Nebula 02 时代只有 MSMF 能开、DSHOW 全废；
          换成 USB 2.0 Camera 后 DSHOW 能开、MSMF 连构造函数都会阻塞。
        所以顺序只作"优先猜测"，真正决定用哪个必须靠能否读到有效帧。
      - 分辨率：不显式设置时 OpenCV 往往只给 640x480；但设了**有时反而打不开流**。
        所以先按请求分辨率试一轮，全失败再退到"不设分辨率"兜底一轮。

    ⚠️ 三层坑，都在这里兜住：
      ① **能打开 ≠ 有画面**：开流可能返回 ok=True 但内容全零的占位帧
         （见 frame_is_usable），此时要**换新句柄重开**（见 MAX_OPEN_ATTEMPTS）。
      ② **创建必须在调用线程**：DSHOW 是 COM，cap 绑定创建它的线程；
         在工作线程创建再交给别人用 = 拿到废句柄（实测 0/5，见 OPEN_PROBE_TIMEOUT 的文档）。
      ③ **构造函数可能阻塞**：所以回退到可疑后端前，先用**子进程**探一下
         （可硬杀），避免主进程永久卡死。

    重试按"**轮次**"组织而不是"把某个组合试到底"：每一轮都让所有还活着的
    后端各试一次。这样首选后端每轮都有机会，可疑后端最多各付一次探测代价就被跳过。
    """
    _need_cv()
    bts = _build_backends()
    table = dict(bts)
    bt = [(backend, table[backend])] if backend and backend in table else bts
    indices = [idx] if idx is not None else [0, 1, 2, 3]
    w_req = width or DEFAULT_CAM_W
    h_req = height or DEFAULT_CAM_H

    blank_tried = []        # 「能打开但画面无效」的组合，用于最后把报错说清楚
    skipped = set()         # 子进程判定不可用 / 会阻塞的后端
    absent = set()          # (后端, 索引) 确认设备不在，不必反复开
    preferred = bts[0][0] if bts else None   # 首选后端：直接试，不做子进程探测

    for res in ((w_req, h_req), (None, None)):
        for attempt in range(MAX_OPEN_ATTEMPTS):
            for name, be in bt:
                if name in skipped:
                    continue
                for i in indices:
                    if (name, i) in absent:
                        continue
                    # 非首选后端先做子进程可用性探测（防构造函数卡死）
                    if name != preferred:
                        if not backend_viable_subprocess(
                                name, i, res[0] or w_req, res[1] or h_req):
                            blank_tried.append(
                                (name, i, res, {"max_pixel": -1, "timed_out": True}))
                            skipped.add(name)
                            break
                    # ★ 在**调用线程**里创建（见 OPEN_PROBE_TIMEOUT 的 2x2 实测）
                    cap = cv2.VideoCapture(i, be)
                    if not cap.isOpened():
                        cap.release()
                        absent.add((name, i))
                        continue
                    if res[0]:
                        # ⚠️ FOURCC 必须先于分辨率设置 —— 格式协商发生在选模式时。
                        #    MJPG 是 USB 2.0 上跑高分辨率的唯一出路（YUY2 未压缩
                        #    在 720p 就开始挤带宽极限，开流成功率掉到 60%）。
                        #    驱动不支持该档的 MJPG 时会静默忽略，属正常。
                        cap.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
                    # ⚠️ 必须判「有效帧」，不能只判 ok（见 frame_is_usable 文档）
                    #    预热容限按档位给（见 warmup_budget），否则要么误杀好开流、
                    #    要么在坏开流上白等。
                    ok, frame, stat = wait_usable_frame(
                        cap, blank_abort=warmup_budget(*res) if res[0] else None)
                    if ok:
                        h, w = frame.shape[:2]
                        note = ""
                        if res[0] and (w, h) != (w_req, h_req):
                            note = f"（请求 {w_req}x{h_req}，该组合不支持，已用实际值）"
                        elif res[0] is None:
                            note = "（请求分辨率会打不开流，已退回默认）"
                        retry = f"（第 {attempt + 1} 轮开流才拿到画面）" if attempt else ""
                        print(f"[摄像头] 后端={name} 索引={i} 分辨率={w}x{h}{note}{retry}")
                        return cap, i
                    cap.release()
                    if stat["timed_out"]:
                        # read 层面阻塞，同样认定该后端不可用
                        blank_tried.append((name, i, res,
                                            {"max_pixel": -1, "timed_out": True}))
                        skipped.add(name)
                        break
                    # 全零帧 -> 记下来；还会在下一轮换新句柄重开
                    if attempt == MAX_OPEN_ATTEMPTS - 1:
                        blank_tried.append((name, i, res, dict(stat)))
            if not [(n, b) for (n, b) in bt if n not in skipped]:
                break
        # end attempts
    # 报错要把"打开成功但画面无效"的情况单独说清楚 —— 这跟"打不开"是两种病，
    # 处置完全不同（一个是占用/驱动，另一个是流开了但拿不到数据）。
    hint = ""
    if blank_tried:
        lines = []
        seen = set()
        for nm, i, res, stat in blank_tried:
            rs = f"{res[0]}x{res[1]}" if res and res[0] else "默认"
            key = (nm, rs, bool(stat.get("timed_out")))
            if key in seen:
                continue
            seen.add(key)
            if stat.get("timed_out"):
                why = "该后端阻塞/不可用（已用子进程探测排除）"
            elif stat["max_pixel"] == 0:
                why = "画面全零（流开了但拿不到数据）"
            else:
                why = f"画面异常（最大像素 {stat['max_pixel']}）"
            lines.append(f"    后端={nm} 索引={i} 请求={rs} -> {why}")
        hint = ("\n\n⚠️ 以下组合**能打开设备但读不到有效画面**：\n"
                + "\n".join(lines[:8])
                + f"\n  若为「画面全零」：已自动重开 {MAX_OPEN_ATTEMPTS} 轮仍失败，"
                  "说明该档位确实取不到画面，\n"
                  "    换低一档（如 640x480）再试 —— 用 `--list-cameras` 看哪些档位 ✅。"
                  "\n  若为「该后端阻塞/不可用」：换 `--backend DSHOW` 试另一个后端。")

    sys.exit(
        "打不开任何摄像头（没有能读到**有效画面**的组合）。排查顺序：\n"
        "  1) 分辨率档位是否超出设备能力 —— 换 `--width 640 --height 480` 试\n"
        "  2) 是否被别的程序占用（相机 App / 会议软件 / 浏览器标签页）\n"
        "  3) 隐私设置是否禁止桌面应用访相机\n"
        "     Windows 设置 > 隐私和安全性 > 相机 > 允许桌面应用访问\n"
        "  4) USB 线接触不良 / 供电不足（换口试，优先主板直连口）\n"
        "  5) 用 --list-cameras 看系统到底认到了什么（它会标注每档是否真有画面）"
        + hint
    )


def list_cameras(width=None, height=None):
    """列出所有能打开的组合，并**逐档位标注是否真有画面**。

    ⚠️ 2026-09-28 重写，两个原因：
      1) 原实现只判 `cap.read()` 的 ok，会把"能打开但只返回全零占位帧"的
         档位报成「✅ 实际 1920x1080」—— 实测这颗 USB 2.0 摄像头在 1080p
         下正是如此，排查时被这个假 ✅ 带偏，误以为摄像头正常。
      2) 原实现慢到不可用（一次枚举 200s+ 还没输出）。

    现在：每档独立开合设备（同进程连续 set 会互相污染），逐档验证**有效画面**，
    连续多帧全零则提前判死，并且所有输出立即 flush（否则重定向到文件时看不到）。
    """
    _need_cv()

    def say(s=""):
        print(s, flush=True)

    say("枚举摄像头：后端 × 索引 × 分辨率档位（每档都验证是否真有画面）")
    say("说明：⚠️全零 = 该档位固件不支持；超时 = 该后端不可用；打不开 = 设备不在")
    say()

    # 待测档位：用户显式指定则只测它，否则按像素数自高到低扫一遍
    if width and height:
        tiers = [(width, height)]
    else:
        tiers = [(3840, 2160), (2560, 1440), (1920, 1080), (1600, 1200),
                 (1280, 960), (1280, 720), (1024, 768), (800, 600), (640, 480)]

    usable = []
    dead_backends = set()
    preferred = _build_backends()[0][0] if _build_backends() else None
    for name, be in _build_backends():
        if name in dead_backends:
            say(f"  后端={name}  跳过（前面已测出该后端会阻塞）")
            continue
        # ⚠️ 非首选后端先做子进程可用性探测。
        #    本机 MSMF 的构造函数会**无限阻塞**，直接开会让 --list-cameras
        #    永久挂住 —— 而它正是文档里推荐的第一步排查手段，挂住最要命。
        #    子进程可被硬杀，见 backend_viable_subprocess 的文档。
        if name != preferred:
            probe = backend_viable_subprocess(name, 0, tiers[0][0], tiers[0][1])
            if not probe:
                say(f"  后端={name}  跳过（子进程探测不可用/会阻塞）")
                dead_backends.add(name)
                say()
                continue
        any_index_open = False
        for i in range(3):
            printed_head = False
            for (w, h) in tiers:
                cap = cv2.VideoCapture(i, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                if not printed_head:
                    say(f"  后端={name}  索引={i}")
                    printed_head = True
                    any_index_open = True
                # 先 MJPG 后分辨率（同 open_camera；驱动不支持的档会静默忽略）
                cap.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                ok, frame, stat = wait_usable_frame(
                    cap, tries=3, timeout=2.0, blank_abort=warmup_budget(w, h))
                if ok:
                    ah, aw = frame.shape[:2]
                    note = "" if (aw, ah) == (w, h) else f"（实得 {aw}x{ah}）"
                    say(f"    {w:>4}x{h:<4} ✅ 有画面  最大像素={stat['max_pixel']:>3}{note}")
                    usable.append((name, i, aw, ah))
                elif stat["timed_out"]:
                    say(f"    {w:>4}x{h:<4} ⚠️ 后端阻塞超时（该后端不可用，跳过）")
                    cap.release()
                    dead_backends.add(name)
                    break
                elif stat["max_pixel"] == 0:
                    say(f"    {w:>4}x{h:<4} ⚠️ 全零（该档位固件不支持）")
                elif stat["max_pixel"] > 0:
                    say(f"    {w:>4}x{h:<4} ⚠️ 画面异常（最大像素 {stat['max_pixel']}）")
                else:
                    say(f"    {w:>4}x{h:<4} ⚠️ 读不到帧（可能被占用）")
                cap.release()
            if printed_head:
                say()
        if not any_index_open:
            say(f"  后端={name}  未枚举到设备")
            say()

    if not usable:
        say("未发现任何能读到**有效画面**的组合。")
        say("按上面的失败类型排查：全零=换档位；超时=换后端；打不开=查占用与隐私设置。")
        return

    best = max(usable, key=lambda x: x[2] * x[3])
    say(f"可用组合 {len(usable)} 个（同一摄像头被多后端枚举到属正常）。")
    say(f"最高可用档位：{best[2]}x{best[3]}（后端={best[0]} 索引={best[1]}）")
    if best[2] * best[3] < 1280 * 720:
        say("⚠️ 没能拿到 720p。低分辨率下 1~1.5m 机位的关键点会明显抖动，")
        say("   建议把机位拉近，或换一颗支持 720p 以上的摄像头。")
    say()
    say("在 --camera 参数里用索引即可；若同一索引不同后端行为不一致，用 --backend 指定。")
    say(f"本项目默认档位由 DEFAULT_CAM_W/H 决定，当前为 {DEFAULT_CAM_W}x{DEFAULT_CAM_H}；")
    say("若默认档位不在上面的 ✅ 列表里，用 --width/--height 显式指定一个 ✅ 的值。")


# ---------------------------------------------------------------- 正面指标

def compute_metrics(kps, w=None, h=None):
    """从 33 个关键点算正面视角可用的坐姿指标。kps 为归一化坐标。

    正面指标全部**按肩宽归一化**，所以宽高比的影响在「当前值 / 基线」这一步自然抵消
    （基线与当前值取自同一颗摄像头、同一画幅），无需额外校正。w/h 仅为接口统一，可忽略。

    返回 None 表示「这一帧的指标不可信，应当丢弃」。
    """
    ear_y = (kps[L_EAR].y + kps[R_EAR].y) / 2.0
    sh_y = (kps[L_SH].y + kps[R_SH].y) / 2.0
    sh_x = (kps[L_SH].x + kps[R_SH].x) / 2.0

    shoulder_span = abs(kps[L_SH].x - kps[R_SH].x)  # 肩宽（近似头肩尺度）
    if shoulder_span < 1e-6:
        return None

    m = {
        # 耳到肩的垂直距离 / 肩宽：低头/趴下时变小。
        # 除以肩宽是为了抗「离摄像头远近」——靠近时整体等比放大，比值不变。
        "head_drop": max(0.0, sh_y - ear_y) / shoulder_span,
        # 双耳高度差：歪头时变大
        "head_tilt": abs(kps[L_EAR].y - kps[R_EAR].y) / shoulder_span,
        # 双肩高度差：歪肩时变大
        "shoulder": abs(kps[L_SH].y - kps[R_SH].y) / shoulder_span,
        # 肩宽绝对值：靠近摄像头时变大（距离过近）
        "span": shoulder_span,
        # 头部横向偏移：身体左右倾
        "head_off": abs(((kps[L_EAR].x + kps[R_EAR].x) / 2.0) - sh_x) / shoulder_span,
    }

    # ---- 合理性粗筛：排除检测器误检产生的垃圾关键点 ----
    # 实测踩过：画面里没有人时，检测器偶尔误报，给出的关键点挤成一团
    # （肩宽仅占画面 3.7%、耳肩距是肩宽的 2.66 倍），却被当成有效样本收进基线，
    # 导致"画面无人也能标定成功"这种荒谬结果。必须在这里挡掉。
    if not (SANE["span_min"] <= m["span"] <= SANE["span_max"]):
        return None
    if not (SANE["head_drop_min"] <= m["head_drop"] <= SANE["head_drop_max"]):
        return None
    return m


def build_baseline(samples):
    """样本 -> 基线（取中位数，抗单帧抖动）。容忍各样本键不一致（髋部可能被桌面遮挡）。"""
    keys = set()
    for s in samples:
        keys |= set(s.keys())
    base = {}
    for k in keys:
        vals = [s[k] for s in samples if k in s]
        if vals:
            base[k] = statistics.median(vals)
    return base


def judge(cur, base):
    """正面视角判定。返回违规问题列表。"""
    issues = []
    b = base["head_drop"]
    if b > 1e-6 and (b - cur["head_drop"]) / b > THRESHOLDS["head_drop"]:
        issues.append("低头")
    b = base["head_tilt"]
    if cur["head_tilt"] - b > THRESHOLDS["head_tilt"]:
        issues.append("歪头")
    b = base["shoulder"]
    if cur["shoulder"] - b > THRESHOLDS["shoulder"]:
        issues.append("歪肩")
    b = base["span"]
    if b > 1e-6 and (cur["span"] - b) / b > THRESHOLDS["too_close"]:
        issues.append("太近")
    return issues


# ------------------------------------------------- 侧面视角
#
# 侧面视角专治正面测不了的问题：背部/颈部的前后关系（CVA 颅椎角）。
#
# ⚠️ 与正面最根本的差别：侧面下左右肩在画面里几乎重合，**肩宽不能作为归一化尺度**，
#    所以侧面指标一律用「角度（度）」——角度天然与距离、分辨率无关。
#    而角度必须在**像素空间**里算，归一化坐标会被宽高比压扁（见文件头说明）。
#
# ⚠️ 躯干倾角需要髋部可见，但孩子坐着写作业时髋部常被桌面遮挡，
#    所以 trunk_angle 是可选的（拿不到就跳过，不影响其它判定）。这是设计如此，不是 bug。


def _vis(k):
    v = getattr(k, "visibility", None)
    return v if v is not None else 1.0


def _pick_side(kps):
    """侧面视角下取更可信的一侧（左右选可见度高的那侧）。"""
    best, best_v = (L_EAR, L_SH, L_HIP), -1.0
    for ear_i, sh_i, hip_i in ((L_EAR, L_SH, L_HIP), (R_EAR, R_SH, R_HIP)):
        v = min(_vis(kps[ear_i]), _vis(kps[sh_i]))
        if v > best_v:
            best, best_v = (ear_i, sh_i, hip_i), v
    return best


def _yaw_gain():
    """偏航角补偿系数 1/sin(yaw)：把被压缩的前后位移还原回矢状面真实量。"""
    s = math.sin(math.radians(max(1.0, min(90.0, SIDE_YAW_DEG))))
    return 1.0 / max(s, 1e-6)


def compute_side_metrics(kps, w, h):
    """侧面视角指标。返回 None 表示该帧不可用。

    坐标必须先由归一化转成像素（`* w` / `* h`）再算角度——这是宽高比校正的关键一步。
    """
    if not w or not h:
        return None
    ear_i, sh_i, hip_i = _pick_side(kps)
    if min(_vis(kps[ear_i]), _vis(kps[sh_i])) < 0.5:
        return None

    ear, sh, hip = kps[ear_i], kps[sh_i], kps[hip_i]
    ex, ey = ear.x * w, ear.y * h
    sx, sy = sh.x * w, sh.y * h

    dx = ex - sx      # 头部相对肩的水平偏移（像素，正 = 头在前方）
    dy = sy - ey      # 耳在肩上方的垂直距离（像素，正 = 头在肩上方）
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    gain = _yaw_gain()
    dx_sag = abs(dx) * gain   # 还原到矢状面的真实前后位移

    if dy <= 1e-6:
        # 头低到肩以下（现实几乎不可能，多半是关键点错乱）-> 记为超限，交给下面筛掉
        neck_angle = 90.0
    else:
        neck_angle = math.degrees(math.atan2(dx_sag, dy))

    m = {
        # 颈部前倾角 = 肩 -> 耳 连线与竖直方向的夹角。低头与头前伸都会让它变大，
        # 侧面无法把这两者拆开（都在矢状面内），但提醒层面不需要区分。
        "neck_angle": neck_angle,
        # 肩-耳直线距离：塌腰/趴桌时上身前塌，这个距离明显收缩。
        "sh_ear_len": math.hypot(dx_sag, dy),
    }

    # 躯干倾角 = 髋 -> 肩 连线与竖直方向的夹角。髋不可见就跳过（桌面遮挡很常见）。
    if _vis(hip) >= 0.5:
        hp_x, hp_y = hip.x * w, hip.y * h
        # ⚠️ 髋在肩**下方**（画面 y 更大），所以取绝对值，别写成 sy - hp_y（会恒为负）
        tdx_sag, tdy = abs(sx - hp_x) * gain, abs(hp_y - sy)
        if tdy > 1e-6:
            m["trunk_angle"] = math.degrees(math.atan2(tdx_sag, tdy))

    # ---- 合理性粗筛（同 compute_metrics 的思路：宁松勿严，只挡明显的误检） ----
    if not (0.0 <= m["neck_angle"] <= SIDE_SANE["neck_angle_max"]):
        return None
    if m["sh_ear_len"] < SIDE_SANE["len_frac_min"] * min(w, h):
        return None
    if "trunk_angle" in m and not (0.0 <= m["trunk_angle"] <= SIDE_SANE["trunk_angle_max"]):
        return None
    return m


def judge_side(cur, base):
    """侧面视角判定。返回违规问题列表。

    三个指标互相独立，缺哪个跳过哪个（髋被遮挡时就没有 trunk_angle）。
    """
    issues = []

    b = base.get("neck_angle")
    if b is not None and cur.get("neck_angle", b) - b > SIDE_THRESHOLDS["neck_angle"]:
        issues.append("头前倾")

    b = base.get("trunk_angle")
    if b is not None and "trunk_angle" in cur and \
            cur["trunk_angle"] - b > SIDE_THRESHOLDS["trunk_angle"]:
        issues.append("塌腰")

    # 肩-耳距离收缩：低头伏案 / 趴桌。注意方向是「变小」才算异常。
    b = base.get("sh_ear_len")
    if b and b > 1e-6 and (b - cur.get("sh_ear_len", b)) / b > SIDE_THRESHOLDS["head_collapse"]:
        issues.append("低头")

    return issues


def fuse(front_issues, side_issues):
    """双机位融合判定——这是双机位相对单路的核心价值。

    两路各自看到的证据相互印证时，可以给出置信度更高的结论：
      正面看到「低头」+ 侧面看到「头前倾」 -> 不是单纯的低头，而是「含胸驼背」。
    单路时这个判断做不出来（正面看不到前后关系，侧面看不出左右对称性）。

    纯侧面模式（--side-only）不调用本函数，所有结论都退回单项。
    """
    merged = list(front_issues) + list(side_issues)
    if "低头" in merged and "头前倾" in merged:
        merged = [i for i in merged if i not in ("低头", "头前倾")]
        merged.insert(0, "含胸驼背")   # 两路互证，升级为更强的结论
    return merged


# ---------------------------------------------------------------- 阈值标定辅助
#
# 背景：原型里的阈值是**未标定的初始值**，必须用真实数据回归，否则误报率不可控。
#
# 但"录 15 分钟视频再逐帧人工标注"这条路走不通——家长不会做，数据收不上来。
# 所以改成：提醒响了就自动存一张现场图，家长事后顺手点一下"对/不对"。
# 这批人工标注就是回归阈值的数据源，而误报率（验收标准 <10%）也只有它能算出来。


def side_score(metric, metrics, base):
    """把侧面指标换算成「越大越该报警」的统一分数。

    ⚠️ 几个指标的方向**不一致**，必须统一，否则扫阈值会反：
      - neck_angle / trunk_angle：比基线**大**才异常（角度增量为正）
      - head_collapse         ：比基线**小**才异常（肩耳距离收缩）
    统一后的语义：`score > 阈值` 即报警，与 judge_side 的判法一致。

    数据不足（如髋部不可见导致没有 trunk_angle）返回 None。
    """
    if metric in ("neck_angle", "trunk_angle"):
        cur, b = metrics.get(metric), (base or {}).get(metric)
        if cur is None or b is None:
            return None
        return float(cur) - float(b)
    if metric == "head_collapse":
        cur, b = metrics.get("sh_ear_len"), (base or {}).get("sh_ear_len")
        if cur is None or b is None or float(b) <= 1e-6:
            return None
        return (float(b) - float(cur)) / float(b)
    return None


def suggest_threshold(scores_should, scores_shouldnt, current=None, fp_weight=1.5):
    """在两组分数之间找最佳阈值。规则：`score > 阈值` 即报警。

    scores_should   : 人工审核认为**确实该报警**的样本分数
    scores_shouldnt : 人工审核认为**不该报警**（误报）的样本分数
    fp_weight       : 误报的权重。默认 1.5 —— **误报比漏报致命**：
                      叫得太勤，孩子三天就把程序关了（任务书 §6）。

    返回 dict；样本不足（两类各需至少 1 条）返回 None。
    """
    a = [float(x) for x in scores_should if x is not None]
    b = [float(x) for x in scores_shouldnt if x is not None]
    if not a or not b:
        return None

    def cost(t):
        fp = sum(1 for v in b if v > t)      # 不该报却报了
        fn = sum(1 for v in a if v <= t)     # 该报却没报
        return fp * fp_weight + fn, fp, fn

    # 候选阈值取所有样本值的中间点（含略低于最小值，表示"几乎不报警"）
    vals = sorted(set(a + b))
    cands = [vals[0] - 1e-6]
    cands += [(vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1)]
    cands += [vals[-1] + 1e-6]

    best = min(cands, key=lambda t: (cost(t)[0], cost(t)[1]))   # 同分时优先少误报
    _, fp_new, fn_new = cost(best)

    def fp_at(t):
        return sum(1 for v in b if v > t)

    return {
        "suggested": best,
        "current": current,
        "n_should": len(a),
        "n_shouldnt": len(b),
        "fp_at_suggested": fp_new,
        "fn_at_suggested": fn_new,
        "fp_at_current": None if current is None else fp_at(float(current)),
        "score_min_should": min(a),
        "score_max_shouldnt": max(b),
    }


# ---------------------------------------------------------------- 迟滞判定
#
# ⚠️ 这段逻辑**必须只有一份实现**。
# 早期它内联在 CLI 主循环里；加图形界面时如果照着再写一遍，
# 「连续 4 秒才提醒」这套规则会在两条路径上慢慢漂移（一处改了另一处忘），
# 而它恰好是最容易毁掉产品的参数（误报 → 孩子三天就把程序关了）。
# 所以抽成 ReminderGate，CLI 与 GUI 共用。


class GateState:
    """一帧的迟滞判定结果。"""

    __slots__ = ("issue", "elapsed", "confirmed", "fire", "cleared")

    def __init__(self, issue=None, elapsed=0.0, confirmed=False, fire=None, cleared=False):
        self.issue = issue        # 当前持续中的问题（未确认时是"观察中"的候选）
        self.elapsed = elapsed    # 该问题已持续多少秒
        self.confirmed = confirmed  # 是否已越过 sustain 门槛（越过了才算"确实歪了"）
        self.fire = fire          # 本次要播报的问题（受 remind_gap 限制），平时为 None
        self.cleared = cleared    # 本次是否刚刚解除（可用于"坐正了，真棒"的正反馈）

    def __repr__(self):
        return (f"GateState(issue={self.issue!r}, elapsed={self.elapsed:.1f}, "
                f"confirmed={self.confirmed}, fire={self.fire!r}, cleared={self.cleared})")


class ReminderGate:
    """迟滞 + 提醒间隔控制。

    规则（防"抖一下就叫"的三道闸）：
      1. 单帧越界**绝不**报警，需连续超标 `sustain` 秒；
      2. 恢复正常后需持续 `clear` 秒才解除；
      3. 同类问题两次播报至少间隔 `gap` 秒。

    ⚠️ 这三个值不要为了"更灵敏"调小。
    """

    def __init__(self, sustain=SUSTAIN_SEC, clear=CLEAR_SEC, gap=REMIND_GAP_SEC,
                 any_gap=REMIND_GAP_ANY_SEC):
        self.sustain = float(sustain)
        self.clear = float(clear)
        self.gap = float(gap)             # 同类问题的最小间隔
        self.any_gap = float(any_gap)     # 任意两次提醒之间的最小间隔（**跨问题也生效**）
        self.issue = None
        self.since = None
        self.good_since = None
        self.last_fired = {}
        self.last_any = None

    def reset(self):
        self.issue = self.since = self.good_since = None
        self.last_fired.clear()
        self.last_any = None

    def update(self, issues, now):
        """issues: 当前帧判出的问题列表；now: 单调时钟秒数。返回 GateState。"""
        if issues:
            self.good_since = None
            cand = issues[0]
            if self.issue != cand:
                # 问题变了 -> 观察窗口重新计时（换了问题就不算"持续"）
                self.issue, self.since = cand, now
            elif now - self.since >= self.sustain:
                same_ok = now - self.last_fired.get(cand, -1e18) >= self.gap
                any_ok = self.last_any is None or now - self.last_any >= self.any_gap
                if same_ok and any_ok:
                    self.last_fired[cand] = now
                    self.last_any = now
                    return GateState(cand, now - self.since, True, cand, False)
                return GateState(cand, now - self.since, True, None, False)
            return GateState(cand, now - self.since, False, None, False)

        # 本帧没判出问题
        cleared = False
        if self.issue:
            if self.good_since is None:
                self.good_since = now
            elif now - self.good_since >= self.clear:
                self.issue = self.since = self.good_since = None
                cleared = True
        else:
            self.good_since = now
        # ⚠️ 注意这里返回的是 `self.issue` 而**不是** None。
        # 恢复期内 self.issue 仍在（等 clear 秒数走完），界面就应当继续显示这个问题，
        # 否则孩子稍微坐正一下就立刻显示"坐得真棒"、下一秒又变回警示 —— UI 会闪，
        # 而 clear 迟滞存在的全部意义正是防这个。
        return GateState(self.issue, 0.0, False, None, cleared)


# ---------------------------------------------------------------- 提醒

VOICE_DIR = os.path.join(HERE, "voice")


def play_reminder(issue):
    """播放提醒音：优先预生成的语音 wav，缺失则退回系统提示音。

    ⚠️ 这里**只管播放**，不做间隔控制——间隔由 ReminderGate 负责（见上）。
    早期版本在播放函数里也做了一次 gap 判断，导致同一规则散落在两处。
    """
    wav = os.path.join(VOICE_DIR, f"{issue}.wav")
    if os.path.isfile(wav):
        try:
            import winsound
            winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
            print(f"  >>> 语音提醒：{issue}")
            return True
        except Exception:
            pass
    try:
        import winsound
        winsound.Beep(880, 180)
    except Exception:
        print("\a", end="")
    print(f"  >>> 提醒：{issue}")
    return False


# ---------------------------------------------------------------- 基准模式

def bench(args):
    """测本机推理性能——这是「本机能否本地胜任」的决定性测试。"""
    _need_cv()
    _need_mp()
    path = find_model(args.model)
    cap, _ = open_camera(args.camera, args.backend, args.width, args.height)
    print("[基准] 预热中 ...")
    for _ in range(5):
        read_frame(cap)

    base = mp_python.BaseOptions(model_asset_path=path)
    opts = vision.PoseLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO, num_poses=1
    )
    times, hit_t, miss_t, hits = [], [], [], 0
    N = args.bench_frames
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        t_start = time.perf_counter()
        for i in range(N):
            ok, frame = read_frame(cap)
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            res = lm.detect_for_video(mi, int((time.perf_counter() - t_start) * 1000))
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            if res.pose_landmarks:
                hit_t.append(dt)
                hits += 1
            else:
                miss_t.append(dt)
    cap.release()

    def stat(a):
        if not a:
            return "无样本"
        a = sorted(a)
        return (f"n={len(a):3d}  mean={statistics.mean(a):6.1f}ms  "
                f"p95={a[min(len(a) - 1, int(len(a) * 0.95))]:6.1f}ms")

    rate = hits / max(1, len(times))
    mean_all = statistics.mean(times)
    mean_hit = statistics.mean(hit_t) if hit_t else None
    mean_miss = statistics.mean(miss_t) if miss_t else None

    print(f"\n===== 基准结果（{os.path.basename(path)}）=====")
    print(f"  采样帧数     : {len(times)}")
    print(f"  检出成功率   : {hits}/{len(times)}  ({rate*100:.0f}%)")
    print()
    print(f"  【含人帧】   {stat(hit_t)}   <- 这才是真实成本")
    print(f"  【无人的帧】 {stat(miss_t)}")

    # MediaPipe 流水线 = 检测器找人框 -> 找到才跑关键点模型，所以两段成本可以相减拆开。
    # ⚠️ 但前提是「含人帧」确实含人。检出率低时那些帧多半是误检，
    #    「有人」样本本身不可信，拆出的比例也是假的 —— 实测踩过：
    #    误检帧给出「关键点仅占 7%」，而真人在图里的干净测量是 29%。差 4 倍。
    if rate < 0.5:
        print("\n  [成本拆解] 跳过 —— 检出率过低，『含人帧』可能多为误检，拆解结果不可信。")
    elif mean_hit and mean_miss:
        kp_share = (mean_hit - mean_miss) / mean_hit * 100
        print(f"\n  [成本拆解] 检测器 ≈ {mean_miss:.1f}ms（{100-kp_share:.0f}%），"
              f"关键点模型 ≈ {mean_hit-mean_miss:.1f}ms（{kp_share:.0f}%）")
        print(f"             -> ROI 跟踪跳过全图检测器可省掉其中约 {100-kp_share:.0f}%")

    # 关键防坑：检出率偏低时绝不能拿总均值下结论
    if not hit_t:
        print("\n  ⚠️ 整段采样一帧都没检测到人。")
        print("     此时跑的只是人体检测器，关键点模型全程未运行 ——")
        print("     本结果**不能代表真实推理成本**（会偏乐观约 40%）。")
        print("     请让一个人完整坐在摄像头前重跑。")
        return

    ref = mean_all
    if rate < 0.5:
        print(f"\n  ⚠️ 检出率仅 {rate*100:.0f}%，以下『总均值』不可用于结论！")
        print(f"     总均值 {mean_all:.1f}ms 被 {len(miss_t)} 个『画面里没人』的帧拉低了 "
              f"{(1 - mean_all / mean_hit) * 100:.0f}%（真实含人成本 {mean_hit:.1f}ms）。")
        print("     请让一个人完整坐在摄像头前重跑；本次结论已自动改用【含人帧】计算。")
        ref = mean_hit

    print(f"\n  ---- 结论（按 {ref:.1f}ms / 帧）----")
    print(f"  等效帧率   : {1000/ref:.1f} FPS")
    print(f"  判定需要的帧率仅 {FPS_SAMPLE} FPS")
    ratio = (1000 / ref) / FPS_SAMPLE
    verdict = "✅ 完全胜任" if ratio > 3 else ("⚠️ 勉强够用" if ratio > 1.5 else "❌ 建议降级")
    print(f"  余量       : {ratio:.1f}x  ->  {verdict}")
    print("\n把以上输出发回，即可确认是否需要启用降级阶梯。")


# ---------------------------------------------------------------- 主流程

def make_landmarker(model_path):
    """每路摄像头要独立一个 landmarker 实例（VIDEO 模式内部维护时间戳状态）。"""
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )


def grab(cap, lm, t_sec, metrics_fn):
    """读一帧 + 推理。返回 (frame, metrics|None, kps|None)。"""
    ok, frame = read_frame(cap)
    if not ok:
        return None, None, None
    h, w = frame.shape[:2]
    mi = mp.Image(image_format=mp.ImageFormat.SRGB,
                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(mi, int(t_sec * 1000))
    if not res.pose_landmarks:
        return frame, None, None
    kps = res.pose_landmarks[0]
    return frame, metrics_fn(kps, w, h), kps


KEYPOINTS_DRAW = (NOSE, L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP)


def draw(frame, kps, status, color, sub=None):
    h, w = frame.shape[:2]
    if kps is not None:
        for i in KEYPOINTS_DRAW:
            cv2.circle(frame, (int(kps[i].x * w), int(kps[i].y * h)), 4, (0, 255, 0), -1)
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if sub:
        cv2.putText(frame, sub, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame


def build_cameras(args):
    """按模式装配机位：返回 [(cam_idx, metrics_fn, judge_fn, tag), ...]"""
    if args.side_only:
        if args.camera2 is not None:
            sys.exit("--side-only 与 --camera2 互斥：纯侧面模式只需要一路摄像头。\n"
                     "纯侧面请用： --side-only --camera <侧面机位索引>")
        return [(args.camera, compute_side_metrics, judge_side, "侧面")]
    cams = [(args.camera, compute_metrics, judge, "正面")]
    if args.camera2 is not None:
        cams.append((args.camera2, compute_side_metrics, judge_side, "侧面"))
    else:
        print("      提示：驼背/头前倾要靠侧面机位。")
        print("            只有侧面一颗摄像头 -> 用 --side-only --camera <索引>")
        print("            有正+侧两颗        -> 用 --camera <正面> --camera2 <侧面>")
    return cams


def main(args):
    _need_cv()
    _need_mp()

    global SIDE_YAW_DEG
    SIDE_YAW_DEG = args.side_yaw

    path = find_model(args.model)
    print(f"[模型] {path}")
    if args.side_only:
        print(f"[模式] 纯侧面单路（偏航角 {SIDE_YAW_DEG:.0f}°）—— 测头前倾 / 塌腰 / 低头")
        if SIDE_YAW_DEG < 70:
            print(f"      ⚠️ 建议把机位摆到 80~90°：{SIDE_YAW_DEG:.0f}° 下前后位移只能看到 "
                  f"{math.sin(math.radians(SIDE_YAW_DEG))*100:.0f}%，容易漏报。")

    specs = build_cameras(args)
    caps, lms, tags, fns, jfs = [], [], [], [], []

    print("[摄像头]")
    for cam_idx, mfn, jfn, tag in specs:
        cap, idx = open_camera(cam_idx, args.backend, args.width, args.height)
        caps.append(cap)
        fns.append(mfn)
        jfs.append(jfn)
        tags.append(tag)
        print(f"      -> {tag}机位已就绪（索引 {idx}，后端自动探测）")

    dual = len(caps) > 1
    for _ in caps:
        lms.append(make_landmarker(path))

    interval = 1.0 / FPS_SAMPLE
    t0 = time.perf_counter()

    try:
        # ------------------------------ 阶段一：标定
        print(f"\n[标定] 请端坐保持 {args.calib:.0f} 秒，正在采集基线 ...")
        samples = [[] for _ in caps]
        last = [None] * len(caps)
        next_run = t0
        last_log = t0
        timed_out = False

        while True:
            now = time.perf_counter()
            if now >= next_run:
                next_run = now + interval
                for i, cap in enumerate(caps):
                    fr, m, _ = grab(cap, lms[i], now - t0, fns[i])
                    if fr is not None:
                        last[i] = fr
                    if m:
                        samples[i].append(m)
                if now - last_log > 1.0:
                    last_log = now
                    print("  已采集：" + "  ".join(
                        f"{tags[i]}={len(samples[i])}" for i in range(len(caps))))
                if not args.no_window:
                    for i, fr in enumerate(last):
                        if fr is not None:
                            cv2.imshow(f"Posture-{tags[i]}",
                                       draw(fr.copy(), None,
                                            f"CALIB {tags[i]} {len(samples[i])}", (0, 200, 255)))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if min(len(s) for s in samples) >= 15 and now - t0 > args.calib:
                break
            # 兜底：任一路始终看不到人时，不能无限等下去
            if now - t0 > max(args.calib * 4, 60):
                timed_out = True
                break

        if timed_out or min(len(s) for s in samples) < 10:
            print("\n[标定失败] 各路样本数：" +
                  "  ".join(f"{tags[i]}={len(samples[i])}" for i in range(len(caps))))
            sys.exit("请确认每一路机位都拍到了完整上半身、光线充足，然后重试。")

        bases = [build_baseline(s) for s in samples]
        for i in range(len(caps)):
            print(f"[标定完成] {tags[i]} 基线 = " +
                  ", ".join(f"{k}={v:.3f}" for k, v in sorted(bases[i].items())))
        print_sanity(tags, bases)

        # ------------------------------ 阶段二：监测
        print(f"\n[监测] 连续超标 {SUSTAIN_SEC:.0f} 秒才提醒；按 q 退出。")
        if dual:
            print("        融合规则：正面「低头」+ 侧面「头前倾」=> 判为「含胸驼背」")
        else:
            print(f"        单侧面模式：独立判定 " +
                  " / ".join(sorted(("头前倾", "塌腰", "低头"))))
        print()

        gate = ReminderGate()
        frames = 0
        paused = False                 # 空格键切换（与 GUI 同一套交互）
        pause_t0 = 0.0
        t_fps = time.perf_counter()
        next_run = time.perf_counter()
        if not args.no_window:
            print("窗口焦点在视频窗上时：空格 = 暂停 / 继续，q = 退出\n")

        while True:
            now = time.perf_counter()
            if now < next_run:
                if args.no_window:
                    time.sleep(0.003)
                else:
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord(" "):
                        paused = not paused
                        if paused:
                            pause_t0 = time.perf_counter()
                            print("⏸  已暂停（再按空格继续）")
                        else:
                            # ⚠️ 暂停期间 gate 一直没被 update()，恢复时 `now - since`
                            #    会突然变大，可能立刻误报一次 —— 所以必须 reset()。
                            #    MediaPipe 的时间戳不用管：暂停期间不调用推理，
                            #    恢复后 `now - t0` 只是往前跳了一段，仍是单调递增。
                            dt = time.perf_counter() - pause_t0
                            t_fps += dt        # FPS 统计里扣掉暂停时间
                            gate.reset()
                            print("▶  已继续")
                continue
            next_run = now + interval
            frames += 1

            if paused:
                # 暂停：不推理、不判定，但继续读帧刷新画面，否则窗口会「无响应」
                for cap, tag in zip(caps, tags):
                    fr = read_frame(cap)
                    if fr is not None and not args.no_window:
                        cv2.imshow(f"Posture-{tag}",
                                   draw(fr.copy(), None, "PAUSED  (SPACE to resume)"))
                continue

            front_issues, side_issues, views = [], [], []
            for i, cap in enumerate(caps):
                fr, m, kps = grab(cap, lms[i], now - t0, fns[i])
                if fr is None:
                    views.append((None, None, "读帧失败"))
                    continue
                if m is None:
                    views.append((fr, kps, "未检测到人"))
                    continue
                st = jfs[i](m, bases[i])
                if tags[i] == "正面":
                    front_issues = st
                else:
                    side_issues = st
                views.append((fr, kps, "坐姿良好" if not st else "！" + "/".join(st)))

            all_issues = fuse(front_issues, side_issues) if dual else (front_issues or side_issues)

            # 迟滞：连续超标才提醒，持续良好才解除（逻辑在 ReminderGate，与 GUI 共用）
            st = gate.update(all_issues, now)
            if st.fire:
                play_reminder(st.fire)

            if not args.no_window:
                fps_now = frames / max(1e-6, now - t_fps)
                for i, (fr, kps, st) in enumerate(views):
                    if fr is None:
                        continue
                    if st == "坐姿良好":
                        col = (0, 200, 0)
                    elif st.startswith("！"):
                        col = (0, 0, 255)
                    else:
                        col = (0, 200, 255)
                    sub = f"{fps_now:.1f} FPS"
                    if dual and all_issues:
                        sub += "  | 融合: " + "/".join(all_issues)
                    cv2.imshow(f"Posture-{tags[i]}",
                               draw(fr.copy(), kps, f"{tags[i]}: {st}", col, sub))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for c in caps:
            c.release()
        cv2.destroyAllWindows()

    print("[退出] 正常结束。")


def sanity_rows(tags, bases):
    """标定结果的合理性自检——别跳，能挡掉大量"看着跑起来了其实是垃圾基线"的情况。

    返回 [(标签, 值或None, 判定, 说明)]，判定 ∈ {"ok", "warn", "info"}。
    图形界面直接拿它上色，CLI 用 print_sanity 打印。
    """
    rows = []
    for tag, b in zip(tags, bases):
        if tag == "正面":
            v = b.get("span")
            rows.append((f"{tag} · 肩宽 span", v,
                         "ok" if v is not None and 0.15 <= v <= 0.60 else "warn",
                         "合理 0.15~0.60；超出说明机位距离不对或帧被误检"))
            v = b.get("head_drop")
            rows.append((f"{tag} · 耳肩距 head_drop", v,
                         "ok" if v is not None and 0.8 <= v <= 2.0 else "warn",
                         "合理 0.8~2.0；超出说明关键点错乱"))
            v = max(abs(b.get("head_tilt", 0.0)), abs(b.get("shoulder", 0.0)))
            rows.append((f"{tag} · 歪头/歪肩基线", v,
                         "ok" if v < 0.05 else "warn",
                         "应接近 0；明显偏大说明标定时孩子没坐正，重标"))
        else:
            v = b.get("neck_angle")
            rows.append((f"{tag} · 颈部前倾角 neck_angle", v,
                         "ok" if v is not None and 0 <= v <= 20 else "warn",
                         "理想 0~20°；超出说明机位或姿势有问题"))
            v = b.get("trunk_angle")
            if v is None:
                rows.append((f"{tag} · 躯干倾角 trunk_angle", None, "info",
                             "这不是故障：髋部被桌面挡住时拿不到，程序会自动跳过「塌腰」"))
            else:
                rows.append((f"{tag} · 躯干倾角 trunk_angle", v,
                             "ok" if 0 <= v <= 20 else "warn", "理想 0~20°"))
    return rows


def print_sanity(tags, bases):
    print("\n[标定自检]")
    for label, v, verdict, hint in sanity_rows(tags, bases):
        mark = {"ok": "✅", "warn": "⚠️ ", "info": "ℹ️ "}[verdict]
        val = "（本帧拿不到）" if v is None else f"{v:.3f}"
        print(f"  {mark} {label:34s} = {val}")
        if verdict in ("warn", "info"):
            print(f"      {hint}")
    print("  提示：数值明显超出区间 -> 孩子标定时就没坐正，或机位/光照有问题，重新标。")


# ---------------------------------------------------------------- 视频回放（离线复现判定）

class VideoSource:
    """把视频文件伪装成摄像头对象（接口与 cv2.VideoCapture 一致）。

    ⚠️ 它只换掉「帧从哪来」。**判定规则仍是同一份实现**
       （`compute_side_metrics` / `judge_side` / `ReminderGate` / `SIDE_THRESHOLDS`），
       所以回放出来的时间线与实跑一致 —— 这正是当初把规则从界面里抽出来的原因。
    """

    def __init__(self, path):
        if cv2 is None:
            sys.exit("缺少 opencv（cv2），无法回放视频。")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit(f"打不开视频文件：{path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0     # 有些容器报 0/NaN
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.i = 0

    # ---- 与 VideoCapture 兼容的最小接口（read_frame / grab 都能直接用）
    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.i += 1
        return ok, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def set(self, *_a):
        return False                    # 文件没有"设分辨率"这回事

    def get(self, prop):
        return self.cap.get(prop)

    @property
    def vtime(self):
        """已读帧对应的**视频内时间**（秒）。回放的时间轴用它，不用墙上时钟。"""
        return self.i / self.fps

    @property
    def duration(self):
        return (self.frames / self.fps) if self.frames else 0.0


def _mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _fmt_metrics(m, b):
    """把当前指标与基线并排打出来，方便一眼看出"差多少才越界"。"""
    parts = []
    for k in ("neck_angle", "trunk_angle"):
        v = m.get(k)
        if v is None:
            continue
        bv = b.get(k)
        parts.append(f"{k}={v:.1f}°" + (f"(基线 {bv:.1f})" if bv is not None else ""))
    v = m.get("sh_ear_len")
    if v is not None and b.get("sh_ear_len"):
        parts.append(f"肩耳距={v:.0f}(基线 {b['sh_ear_len']:.0f})")
    return "  ".join(parts)


def _load_baseline_json():
    """读 baseline.json 里的 base 字段（与界面共用同一份基线，不含"壳"）。"""
    p = os.path.join(HERE, "baseline.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("base") or None
    except Exception:
        return None


def replay(args):
    """离线回放一段视频，跑完整判定链并输出逐条时间线。

    为什么要有它：判定准不准（误报/漏报）以前只能"当场盯着看"。
    现在录一段视频就能离线重跑整条链路 —— **不占人在场的时间，而且可复现**
    （同一视频 + 同一阈值 → 同一时间线），用来调阈值比看回放录像快得多。

    用法：
        python posture_monitor.py --video rec.mp4 --calib 10   # 用视频前 10 秒标定
        python posture_monitor.py --video rec.mp4 --calib 0    # 用现有 baseline.json

    时间基准是**视频内时间**，不 sleep、不开窗，所以跑得比真实播放快很多。
    """
    _need_cv()
    _need_mp()

    if args.camera2 is not None:
        sys.exit("[回放] --video 目前只支持纯侧面一路（判定规则与 --side-only 相同）；"
                 "请去掉 --camera2。")

    global SIDE_YAW_DEG
    SIDE_YAW_DEG = args.side_yaw

    src = VideoSource(args.video)
    print(f"[回放] {args.video}")
    print(f"        {src.frames} 帧 · {src.fps:.1f} FPS · 时长 {src.duration:.1f}s")
    lm = make_landmarker(find_model(args.model))
    step = max(1, int(round(src.fps / max(1, int(FPS_SAMPLE)))))   # 每 N 帧判定一次

    try:
        # ---------------- 标定：用视频前 calib 秒，或直接用现有 baseline
        if args.calib and args.calib > 0:
            t_end = min(src.duration, float(args.calib))
            samples = []
            while src.i < src.frames and src.vtime < t_end:
                if src.i % step != 0:
                    if not src.read()[0]:
                        break
                    continue
                fr, m, _ = grab(src, lm, src.vtime, compute_side_metrics)
                if fr is None:
                    break
                if m:
                    samples.append(m)
            if len(samples) < 10:
                sys.exit(f"[回放] 标定失败：前 {t_end:.0f}s 只采到 {len(samples)} 个有效样本。\n"
                         f"       检查这段视频里是否有人、光线是否够、机位是否侧面；\n"
                         f"       或改用 --calib 0（走已有的 baseline.json）。")
            base = build_baseline(samples)
            print(f"[回放] 前 {t_end:.0f}s 标定基线：" +
                  ", ".join(f"{k}={v:.2f}" for k, v in sorted(base.items())))
        else:
            base = _load_baseline_json()
            if not base:
                sys.exit("[回放] --calib 0 需要 baseline.json：先用界面标定一次，或改回 --calib 10。")
            print("[回放] 使用现有 baseline.json：" +
                  ", ".join(f"{k}={v:.2f}" for k, v in sorted(base.items())))

        # ---------------- 回放判定（迟滞与提醒间隔仍然走共用 ReminderGate）
        gate = ReminderGate()
        fired, counts = [], {}
        n_judge = n_skip = 0
        prev = []
        print("\n---- 判定时间线 ----")
        while src.i < src.frames - 1:
            if src.i % step != 0:
                if not src.read()[0]:
                    break
                continue
            t = src.vtime
            fr, m, _ = grab(src, lm, t, compute_side_metrics)
            if fr is None:
                break
            if m is None:
                n_skip += 1
                continue
            n_judge += 1
            issues = judge_side(m, base)
            for it in issues:
                counts[it] = counts.get(it, 0) + 1
            st = gate.update(issues, t)
            if issues and issues != prev:
                print(f"[{_mmss(t)}] 判定 → {'/'.join(issues)}    {_fmt_metrics(m, base)}")
            elif not issues and prev:
                print(f"[{_mmss(t)}] 恢复 → 坐姿良好")
            prev = list(issues)
            if st.fire:
                print(f"[{_mmss(t)}] ▶ 提醒「{st.fire}」")
                fired.append((t, st.fire))

        # ---------------- 汇总
        dur = max(1e-6, src.duration)
        print("\n---- 汇总 ----")
        print(f"  时长        {_mmss(src.duration)}")
        extra = f"（另有 {n_skip} 帧未检测到人，已跳过）" if n_skip else ""
        print(f"  判定帧      {n_judge} 帧（每 {step} 帧取 1 帧）{extra}")
        if fired:
            print(f"  提醒        {len(fired)} 次 / {dur / 60:.1f} 分钟"
                  f" → 平均每 {dur / len(fired):.0f} 秒一次")
        else:
            print("  提醒        0 次")
        if counts:
            for it, c in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  问题「{it}」  命中 {c} 个判定帧")
        else:
            print("  问题        没有任何问题命中（是全片都坐得好，还是阈值太松？）")
        print("\n  提示：误报看上面「判定 →」的行与录像是否对得上；"
              "漏报要看有没有该提醒却没出现「▶ 提醒」的片段。")
        return 0
    finally:
        src.release()


# ---------------------------------------------------------------- 入口

def build_parser():
    ap = argparse.ArgumentParser(
        description="坐姿实时提醒 P0 原型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "常用组合：\n"
            "  纯侧面（纸笔作业推荐）  --side-only --camera 0 --calib 10\n"
            "  双机位                  --camera 0 --camera2 1 --calib 10\n"
            "  性能自测（需真人入画）  --bench --bench-frames 100\n"
            "  排查摄像头              --list-cameras\n"
            "  离线复现判定            --video 录像.mp4 --calib 10\n"
        ),
    )
    ap.add_argument("--bench", action="store_true", help="只测推理性能")
    ap.add_argument("--bench-frames", type=int, default=60, help="基准采样帧数")
    ap.add_argument("--list-cameras", action="store_true", help="列出可用摄像头（排查用）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="模型文件名")
    ap.add_argument("--side-only", action="store_true",
                    help="纯侧面单路模式（纸笔作业场景）：只做头前倾/塌腰/低头")
    ap.add_argument("--camera", type=int, default=None,
                    help="主路摄像头索引；--side-only 时指侧面机位")
    ap.add_argument("--camera2", type=int, default=None,
                    help="第二路（侧面）摄像头索引；给了就启用双机位。与 --side-only 互斥")
    ap.add_argument("--side-yaw", type=float, default=90.0,
                    help="侧向机位偏航角（度）：0=正对，90=正侧。默认 90，建议 80~90")
    ap.add_argument("--backend", choices=["DSHOW", "MSMF", "ANY"], default=None,
                    help="强制指定采集后端；默认按 DSHOW->MSMF->ANY 依次尝试")
    ap.add_argument("--width", type=int, default=DEFAULT_CAM_W,
                    help=f"采集宽度，默认 {DEFAULT_CAM_W}（最高清档；本机摄像头固定给 1080p，改小无效）")
    ap.add_argument("--height", type=int, default=DEFAULT_CAM_H,
                    help=f"采集高度，默认 {DEFAULT_CAM_H}")
    ap.add_argument("--calib", type=float, default=10.0, help="标定时长（秒）")
    ap.add_argument("--no-window", action="store_true", help="不开预览窗")
    ap.add_argument("--video", default=None,
                    help="离线回放：给定视频文件，跑完整判定链并输出逐条时间线（不接摄像头）")
    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if _args.list_cameras:
        list_cameras(_args.width, _args.height)
    elif _args.bench:
        bench(_args)
    elif _args.video:
        sys.exit(replay(_args))
    else:
        main(_args)

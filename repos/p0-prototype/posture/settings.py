# -*- coding: utf-8 -*-
"""配置、基线的读写，以及"产品策略默认值"。

分工（重构时最容易分错的一刀）：
    `posture.metrics`  住在**会被运行时改写的判定参数**（阈值、偏航角、合理性区间）
    `posture.gate`     住在**迟滞参数**（连续多久才提醒 / 提醒间隔）
    本模块             住在**产品策略默认值**（采样帧率、默认分辨率）+ **磁盘读写**

为什么迟滞参数不在这：`gate.py` 里 `ReminderGate` 的构造函数默认值直接取自它自己
模块内的常量，放这里反而要反向 import，形成 settings → gate → settings 的环。

★ 分辨率策略（2026-10-18 用户定调）：**坚持 4K 优先，不降级**。
    默认档位就是 3840x2160；开不到 4K 时**不静默换小档**，
    要么按阶梯继续试到 4K，要么把"当前只在 X 档"作为**可见警告**报给用户。
    以前"为了启动快而预先降档"是错的取向 —— 慢可以等，画质不能偷偷变差。

⚠️ 基线（baseline）**必须与采集分辨率绑定**，见 `load_baseline` 的文档。
"""

import json
import os
import time

from posture import gate, metrics, paths


# ---------------------------------------------------------------- 产品策略默认值

FPS_SAMPLE = 5        # 判定采样帧率（推理频率）

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


# ---------------------------------------------------------------- 配置

DEFAULT_CONFIG = {
    "camera": 0,
    "side_only": True,
    "side_yaw": 90.0,
    "model": paths.DEFAULT_MODEL,
    "width": DEFAULT_CAM_W,
    "height": DEFAULT_CAM_H,
    "fps_sample": FPS_SAMPLE,
    "calib_sec": 10.0,
    "mute": False,
    "volume": "中",             # 低 / 中 / 高
    "sustain_sec": gate.SUSTAIN_SEC,
    "clear_sec": gate.CLEAR_SEC,
    "remind_gap_sec": gate.REMIND_GAP_SEC,
    "remind_gap_any_sec": gate.REMIND_GAP_ANY_SEC,
    "side_thresholds": dict(metrics.SIDE_THRESHOLDS),
    "thresholds": dict(metrics.THRESHOLDS),
    # 判定现场留存（供家长审核 —— 这是拿到「误报率 <10%」验收指标的唯一现实途径）
    "review_enabled": True,        # 全部本地存储、不上传；家长可一键清空
    "review_max_files": 300,       # 超出上限自动删最旧的
    "review_sample_sec": 180.0,    # 每隔多久另存一张"判为良好"的抽样（否则测不出漏报）
    # 2026-09-26 用户改口径：摄像头换了高清、并要求**处处都用最高清档**，
    # 所以「提醒现场」的原图**默认就存**（原来为了省空间是关的）。
    # 代价：1080p 原图每张约 150~500KB；且 `_full.jpg` 也计入 review_max_files，
    #       所以实际能留存的**条数**会变少（想省空间就在「家长设置 → 审核判定」关掉）。
    # 剪枝/清空会把原图和缩略图一起处理（见 _files_of），不会留孤儿文件。
    "review_keep_fullres": True,
}


def _read_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None          # 文件坏了就用默认值，不能因此起不来


def load_config(path=paths.CONFIG_PATH):
    cfg = dict(DEFAULT_CONFIG)
    cfg["side_thresholds"] = dict(DEFAULT_CONFIG["side_thresholds"])
    cfg["thresholds"] = dict(DEFAULT_CONFIG["thresholds"])
    saved = _read_json(path)
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in ("side_thresholds", "thresholds") and isinstance(v, dict):
                cfg[k].update({kk: float(vv) for kk, vv in v.items()})
            else:
                cfg[k] = v
    return cfg


def save_config(cfg, path=paths.CONFIG_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def apply_config(cfg):
    """把配置灌进各模块的**模块级全局**参数。

    ⚠️ 判定函数（judge_side 等）读的是模块全局，这是既有设计。
    测试直接调判定函数时用的就是默认值，所以这里做的是"运行时覆盖"，
    而不是改签名——改签名会把几百个回归用例全部牵动（用例数会涨，别在这里写死）。
    """
    global FPS_SAMPLE
    metrics.SIDE_YAW_DEG = float(cfg.get("side_yaw", 90.0))
    FPS_SAMPLE = int(cfg.get("fps_sample", FPS_SAMPLE) or FPS_SAMPLE)
    for k, v in cfg.get("side_thresholds", {}).items():
        metrics.SIDE_THRESHOLDS[k] = float(v)
    for k, v in cfg.get("thresholds", {}).items():
        metrics.THRESHOLDS[k] = float(v)
    return FPS_SAMPLE


# ---------------------------------------------------------------- 基线

def baseline_mismatch(reason, **extra):
    """把"为什么要重新标定"写成人话（界面直接显示，不让用户猜）。"""
    msg = {"camera": "摄像头编号变了",
           "side_yaw": "机位偏航角变了",
           "resolution": "采集分辨率变了",
           "no_resolution": "这份基线没记录采集分辨率",
           "model": "模型档位变了",
           "empty": "还没有标定过"}
    msg.update(extra)
    return msg.get(reason, reason)


def load_baseline(cfg, path=paths.BASELINE_PATH):
    """读回上次的基线，并校验它**是否还配得上当前的采集条件**。

    基线里的 `sh_ear_len` 是**像素**长度，随分辨率等比缩放；
    「低头」判据是 `(基线 - 当前)/基线 > 阈值`。
    ⚠️ 所以基线若是 1080p 标的、现在用 4K 跑，当前值会整整大一倍，
       收缩判据永远不成立 —— **低头这项会静默失效（漏报）**，
       而界面上一切看起来正常。这正是重构前只校验 camera+side_yaw 留下的坑。

    角度类指标（neck_angle / trunk_angle）不受影响，但为了不再有"半可信"的基线，
    分辨率不一致时一律作废、要求重标（10 秒的成本，换掉一个隐形错误）。
    """
    d = _read_json(path)
    if not d:
        return None
    if d.get("camera") != cfg.get("camera"):
        return None
    if abs(float(d.get("side_yaw", -1)) - float(cfg.get("side_yaw", 0))) > 1e-6:
        return None
    # ⚠️ 没记录分辨率的旧基线**也**作废：它可能是上一颗摄像头/另一档分辨率标的，
    #    留着就是上面那种"看着正常其实漏报"的状态。宁可让用户重标一次。
    if d.get("width") is not None and (d.get("width") != cfg.get("width")
                                       or d.get("height") != cfg.get("height")):
        return None
    if d.get("width") is None:
        return None
    return d.get("base") or None


def read_baseline_raw(path=paths.BASELINE_PATH):
    """读 baseline.json 的 `base` 字段，**不做**采集条件校验。

    只给离线回放用：回放是"复现判定"，用户既然指定了 --calib 0 就用现有这份，
    不该拿"分辨率/机位变了"卡住他（回放的帧和实跑本来就不是同一采集条件）。
    实时路径一律走 `load_baseline(cfg)` —— 那条校验是防漏报的，不能绕。
    """
    d = _read_json(path)
    return (d or {}).get("base") or None


def baseline_reason(cfg, path=paths.BASELINE_PATH):
    """给界面用：返回 None（可用）或"为什么要重标"的人话。"""
    d = _read_json(path)
    if not d or not d.get("base"):
        return baseline_mismatch("empty")
    if d.get("camera") != cfg.get("camera"):
        return baseline_mismatch("camera")
    if abs(float(d.get("side_yaw", -1)) - float(cfg.get("side_yaw", 0))) > 1e-6:
        return baseline_mismatch("side_yaw")
    if d.get("width") is None:
        return baseline_mismatch("no_resolution")
    if d.get("width") != cfg.get("width") or d.get("height") != cfg.get("height"):
        return baseline_mismatch("resolution")
    return None


def save_baseline(cfg, base, path=paths.BASELINE_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"camera": cfg.get("camera"),
                       "side_yaw": cfg.get("side_yaw"),
                       # ★ 这三个字段是"基线有效性"的一部分，缺了就必须重标
                       "width": cfg.get("width"),
                       "height": cfg.get("height"),
                       "model": cfg.get("model"),
                       "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "base": base}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

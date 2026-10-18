# -*- coding: utf-8 -*-
"""姿态推理封装（MediaPipe PoseLandmarker）。

全项目只有本模块碰 mediapipe 的**推理 API**（`deps` 只负责导入与可用性）。
这样 SAC/matplotlib 那类"导入期"问题留在 `deps`，
"运行期"问题（时间戳必须单调、VIDEO 模式要独立实例）留在本模块，互不污染。

⚠️ MediaPipe 的两个硬约束（踩过，别改）：
  1. **VIDEO 模式内部维护时间戳状态**：同一实例只接受**单调递增**的时间戳，
     回退会抛 `PERFORMANCE_ERROR` 或返回垃圾结果。
     → 所以**每路摄像头要独立一个 landmarker 实例**；
     → 也所以暂停/恢复时**不能**把时间戳重置回 0（只是跳过一段，仍单调）。
  2. 输入图像固定被下采样到 256x256 再推理 —— **采集分辨率不影响姿态精度**。
     4K 的价值在「提醒现场存证照片清楚」，不在判定。别用"为了准确率要高清"解释它。
"""

from posture import camera, deps, metrics as mx


def make_landmarker(model_path):
    """构造一路 VIDEO 模式姿态推理器。

    ⚠️ 依赖守卫**就放在这里**（而不是散在各个调用方）：
       没有 mediapipe 时抛人话的 SystemExit，
       绝不走到 `vision.PoseLandmarker` 那行变成
       `AttributeError: 'NoneType' object has no attribute 'PoseLandmarker'`
       —— 那个报错和真实病因（缺依赖 / SAC 拦截）毫无关系，害得排查半天。
    """
    deps.need_mp()
    return deps.vision.PoseLandmarker.create_from_options(
        deps.vision.PoseLandmarkerOptions(
            base_options=deps.mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=deps.vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )


def detect(lm, frame, t_sec, metrics_fn):
    """对**已有的一帧**推理并算指标。返回 (指标 dict|None, 关键点|None)。

    `t_sec` 是单调时钟秒（内部乘 1000 转成 MediaPipe 要的毫秒）。
    指标函数 `metrics_fn` 由调用方给定：正面 `mx.compute_metrics`、
    侧面 `mx.compute_side_metrics` —— 本模块不知道也不关心判定规则。
    """
    h, w = frame.shape[:2]
    mi = deps.mp.Image(image_format=deps.mp.ImageFormat.SRGB,
                       data=deps.cv2.cvtColor(frame, deps.cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(mi, int(t_sec * 1000))
    if not res.pose_landmarks:
        return None, None
    kps = res.pose_landmarks[0]
    return metrics_fn(kps, w, h), kps


def grab(cap, lm, t_sec, metrics_fn):
    """读一帧 + 推理。返回 (frame, metrics|None, kps|None)。

    这是 **CLI 的简单路径**（自己拿着 cap 逐帧读）。
    图形界面/引擎不用它 —— 那里读帧归 `camera.CameraStream` 线程管，
    推理拿的是 `stream.latest()` 递过来的帧。
    """
    ok, frame = camera.read_frame(cap)
    if not ok:
        return None, None, None
    mets, kps = detect(lm, frame, t_sec, metrics_fn)
    return frame, mets, kps


# ---------------------------------------------------------------- 画面叠加（CLI 预览窗）

KEYPOINTS_DRAW = (mx.NOSE, mx.L_EAR, mx.R_EAR, mx.L_SH, mx.R_SH, mx.L_HIP, mx.R_HIP)


def draw(frame, kps, status, color, sub=None):
    """把状态文字与关键点画到帧上（只给 CLI 的 OpenCV 窗口用）。

    ⚠️ 图形界面的预览**不经过这里** —— Qt 那边自己画，
       避免"存证照片被叠加了红色警告字"（家长看到会以为孩子真出事了）。
    """
    h, w = frame.shape[:2]
    if kps is not None:
        for i in KEYPOINTS_DRAW:
            deps.cv2.circle(frame, (int(kps[i].x * w), int(kps[i].y * h)),
                            4, (0, 255, 0), -1)
    deps.cv2.putText(frame, status, (20, 40), deps.cv2.FONT_HERSHEY_SIMPLEX,
                     0.9, color, 2)
    if sub:
        deps.cv2.putText(frame, sub, (20, 75), deps.cv2.FONT_HERSHEY_SIMPLEX,
                         0.6, (200, 200, 200), 1)
    return frame


# ---------------------------------------------------------------- 界面预览图

# 预览图宽度上限。
# 界面是按 2x 缩放显示的（本机 devicePixelRatio = 2.00）。
# 实测预览控件约 520 逻辑像素宽 -> 需要 **1040 物理像素**才不虚；
# 取 1280 留出余量（窗口再宽一点也不用上采样）。
# 480 是早期"能看清有没有人"的口径，在 2x 屏上明显偏糊。
# 代价只是每帧多一次 resize（对 5 FPS 可忽略；4K 全管线实测仍只用 1/4 预算）。
# ⚠️ 与界面侧的 ScaledImageLabel 配套：那边在 paintEvent 里按**物理像素**缩放，
#    所以这里的预览图必须**不小于**控件的物理宽度，否则就是上采样发虚。
PREVIEW_MAX_W = 1280


def shrink(frame, max_w=PREVIEW_MAX_W):
    """把帧缩成**界面显示用**的小图。

    ⚠️ 只用于预览。**判定与存图都不走它** —— 判定用原始帧，
       审核原图走 `review.ReviewStore(full_frame=...)`，各自保全最高清。
    """
    if frame is None or deps.cv2 is None:
        return None
    try:
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        k = max_w / float(w)
        return deps.cv2.resize(frame, (max_w, int(h * k)),
                               interpolation=deps.cv2.INTER_AREA)
    except Exception:
        return None

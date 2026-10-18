# -*- coding: utf-8 -*-
"""第三方依赖的**延迟报错**入口。

为什么单独一个模块
------------------
判定逻辑（metrics / gate）是纯数学，离开 cv2 与 mediapipe 也必须能 import ——
`tests/test_offline.py` 就是要在**任何机器**上跑（包括没装依赖的）。
所以全项目的 `import cv2 / import mediapipe` 只写在这一处，别的地方一律
`from posture import deps` 然后用 `deps.cv2`。

★★ 两条不能退让的实测结论（都踩过，改动前先读完）

1. **matplotlib 的桩必须装在 `import mediapipe` 之前。**
   本机 Windows 智能应用控制（SAC）拦掉了 matplotlib/_image.pyd，而 mediapipe 的
   `tasks/python/vision/drawing_utils.py` **无条件** `import matplotlib.pyplot`
   → 整个 vision 包导入失败 → 最后炸成
   `AttributeError: 'NoneType' object has no attribute 'PoseLandmarker'`
   （和真病因毫无关系）。详见 compat_matplotlib 模块文档。

2. **导入异常必须原样留着，不许裸 except 吞掉。**
   早期写的是 `except ImportError: vision = None`，真病因（SAC 拦截）完全看不见，
   排查全靠猜。现在 `_MP_IMPORT_ERR` 一路带到 need_mp() 的报错文案里。
"""

# ---- 先装兼容桩，再 import mediapipe（顺序是硬约束，见上面第 1 条）----
try:
    from posture import compat_matplotlib
    compat_matplotlib.install()
except Exception:
    pass

try:
    import cv2
except ImportError:
    cv2 = None

_MP_IMPORT_ERR = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as _e:
    _MP_IMPORT_ERR = _e
    mp = mp_python = vision = None


def need_cv():
    """需要 cv2 的入口用它把关：缺依赖时报**人话**，而不是走到一半抛 NoneType。"""
    if cv2 is None:
        raise SystemExit("缺少依赖。请先执行： pip install mediapipe opencv-python")


def need_mp():
    if mp is None or vision is None:
        msg = ("缺少 mediapipe（或 mediapipe 导入失败）。"
               "请先执行： pip install mediapipe")
        if _MP_IMPORT_ERR is not None:
            msg += (f"\n\n导入 mediapipe 时的原始报错：\n"
                    f"  {type(_MP_IMPORT_ERR).__name__}: {_MP_IMPORT_ERR}")
            if ("应用程序控制策略" in str(_MP_IMPORT_ERR)
                    or "Smart App Control" in str(_MP_IMPORT_ERR)):
                msg += ("\n\n这看起来是 Windows 智能应用控制（SAC）拦截了第三方扩展 DLL。\n"
                        "  处置顺序：① 先重启一次电脑（实测拦截常会自愈）"
                        " → ② 靠项目内置的 compat_matplotlib 桩兜底"
                        " → ③ 最后才考虑关 SAC（关了不可逆，要恢复得重装 Windows）。\n"
                        "  若仍报此错，说明被拦的不止 matplotlib，"
                        "把上面那段原始报错发出来定位具体文件。")
        raise SystemExit(msg)


def mp_available():
    """给界面/自检用：不抛异常地回答"mediapipe 到底可用不可用"。"""
    return not (mp is None or vision is None)


# 常用常量集中转发，避免各处散写 cv2 的魔法数
MJPG_FOURCC = cv2.VideoWriter_fourcc(*"MJPG") if cv2 is not None else 0
"""UVC 压缩格式。⚠️ 必须**在设置分辨率之前** set —— 格式协商发生在选模式时。
驱动若不提供该档的 MJPG 会静默忽略（退回 YUY2）：USB 2.0 上 YUY2 喂不饱高分辨率，
1080p 档就没有 MJPG，属正常。详见 camera 模块文档。"""

# -*- coding: utf-8 -*-
"""posture 包的路径与运行时目录。

单独一个模块的原因：这些路径以前散在 posture_monitor / posture_engine 里各写一份
（HERE、模型目录、voice 目录、config 路径），重构时最容易漏的就是"两处漂移"。
现在只有这一份，其余模块一律从这里取。

⚠️ ROOT 是**发行目录**（repos/p0-prototype），不是包目录。
config.json / baseline.json / camera_profile.json / voice/ / review/ / logs/ / models/
全在 ROOT 下 —— 用户看得见、`.bat` 也在那里启动，不能把它们挪进包里。
"""

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG_DIR)

MODEL_DIRS = [
    os.path.join(ROOT, "models"),
    os.path.join(ROOT, "..", "..", "bench", "models"),
]
DEFAULT_MODEL = "pose_landmarker_lite.task"

CONFIG_PATH = os.path.join(ROOT, "config.json")
BASELINE_PATH = os.path.join(ROOT, "baseline.json")
CAMERA_PROFILE_PATH = os.path.join(ROOT, "camera_profile.json")

VOICE_DIR = os.path.join(ROOT, "voice")
REVIEW_DIR = os.path.join(ROOT, "review")
LOG_DIR = os.path.join(ROOT, "logs")


def find_model(name=DEFAULT_MODEL):
    """在 MODEL_DIRS 里找模型文件。

    ⚠️ 依赖守卫模块（deps）负责在没有 cv2/mediapipe 时也能 import 本模块，
    所以 find_model 这里**不**碰任何第三方库，纯查文件。
    """
    for d in MODEL_DIRS:
        p = os.path.normpath(os.path.join(d, name))
        if os.path.isfile(p):
            return p
    raise SystemExit(
        f"找不到模型 {name}。\n"
        f"下载地址（注意档位名在路径中出现两次）：\n"
        f"  https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_lite/float16/1/pose_landmarker_lite.task\n"
        f"放到以下任一目录：\n  " + "\n  ".join(os.path.normpath(d) for d in MODEL_DIRS)
    )

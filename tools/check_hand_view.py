#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔视角验证工具
================

**先确认"看得见"，再谈"判得准"。**

握笔姿势检测能不能做，不取决于算法多聪明，而取决于一件事：
摄像头能不能把**手和笔**拍得足够大、足够清楚。

手部关键点模型（MediaPipe Hand Landmarker）在手上只有 **21 个点**，
而且手指节之间的精度有限。实测文献里 MediaPipe 的手部关节角
相对 Vicon 真值的平均误差是 **22.5°**，而"握笔姿势对不对"要靠的
关节角差异往往只有 10~30° —— 误差和信号同一个量级。

在这么紧的余量下，**手在画面里占多少像素**就成了决定性的变量。
手太小，一切都谈不上；手够大，才有讨论的余地。

所以这个工具只干一件事：**告诉你手在画面里有多大、跟踪稳不稳。**

用法
----
    # ① 先拿手机拍一张孩子握笔的照片，零成本试一下（最推荐的第一步）
    python tools/check_hand_view.py --image D:\\握笔照片.jpg

    # ② 实时对着摄像头调机位
    python tools/check_hand_view.py --camera 0

    # ③ 同时显示身体姿态（用来直观看到"一个摄像头兼顾不了"）
    python tools/check_hand_view.py --camera 0 --with-pose

    # ④ 不带摄像头，只验证模型和代码能跑
    python tools/check_hand_view.py --selftest

实时模式按键：
    s  存一张当前画面（存到 hand_view/ 目录，方便事后对比机位）
    q / ESC  退出（会打印一份小结）

⚠️ 两个实现细节（踩过）
------------------------
1. **OpenCV 的 putText 写不了中文**（默认字体不含 CJK，会变问号）。
   所以画面上的字全用英文，中文结论走控制台输出。
2. **MediaPipe 的 detect_for_video 要求时间戳全生命周期单调递增**。
   本项目里为这个坑返工过两次，这里用一个全局计数器保证。
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.normpath(os.path.join(HERE, "..", "repos", "p0-prototype"))
sys.path.insert(0, PROTO)

import posture_monitor as pm   # noqa: E402  （复用相机打开/读帧/模型查找）

HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = pm.DEFAULT_MODEL

# 手部宽度（画面像素）的判定门槛。
# 依据：MediaPipe 的掌部检测器会把 ROI 缩到 224x224，手在画面里太小就抓不稳；
# 而我们要分辨的是**手指之间几十度的形态差异**，需要比"能检测到手"更高的余量。
# ⚠️ 这几个数是工程经验值，不是论文阈值 —— 真正作数的是你肉眼看画面判断。
HAND_W_OK = 200.0      # >= 这个值：够用
HAND_W_MARGINAL = 120.0  # 这个值到 OK 之间：勉强；低于它：太小
WINDOW = 90            # 滑动窗口（约 3 秒 @30fps），用来算检出率与抖动

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),            # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),       # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),     # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),    # 小指
    (0, 17),
]


def make_hand(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,   # 放低一点：宁可检测到再筛，也别漏
        min_tracking_confidence=0.3)
    return vision.HandLandmarker.create_from_options(opts)


def make_pose(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    return vision.PoseLandmarker.create_from_options(opts)


class HandStats:
    """滑动窗口统计：检出率、手部尺寸、抖动。

    为什么要看**检出率**和**抖动**：
      文献里明确记录过 MediaPipe 在遮挡+运动时出现**连续约 4 秒完全返回不了
      有效坐标**的情况（MDPI 的手-物交互研究）。手在写字时时一直在动，
      所以"能不能稳定跟住"和"某一帧跟得准不准"是两件事 —— 判定要做在
      稳定跟住的前提上，否则会疯狂误报。
    """

    def __init__(self, window=WINDOW):
        self.window = window
        self.seen = []        # True/False 每帧是否检测到手
        self.widths = []
        self.ref = []         # 中指根部坐标，用来算抖动

    def add(self, lm, w, h):
        self.seen.append(lm is not None)
        del self.seen[:-self.window]
        if lm is None:
            return
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        self.widths.append(max(xs) - min(xs))
        del self.widths[:-self.window]
        self.ref.append((lm[9].x * w, lm[9].y * h))
        del self.ref[:-self.window]

    @property
    def rate(self):
        return (sum(self.seen) / len(self.seen)) if self.seen else 0.0

    @property
    def mean_w(self):
        return (sum(self.widths) / len(self.widths)) if self.widths else 0.0

    @property
    def jitter(self):
        """中指根部位置的帧间标准差（像素）。越大越不稳。"""
        if len(self.ref) < 3:
            return None
        a = np.array(self.ref)
        return float(np.sqrt(a[:, 0].std() ** 2 + a[:, 1].std() ** 2))

    def verdict(self):
        w = self.mean_w
        if not self.widths:
            return "none", "画面里没检测到手"
        if w >= HAND_W_OK:
            return "ok", f"手部约 {w:.0f}px 宽 —— 尺寸够用"
        if w >= HAND_W_MARGINAL:
            return "marginal", f"手部约 {w:.0f}px 宽 —— 偏小，跟踪会不稳，建议再靠近些"
        return "small", f"手部仅约 {w:.0f}px 宽 —— 太小，这个视角下做不了判定"


def draw_overlay(frame, lm, stats, extra=""):
    import cv2
    h, w = frame.shape[:2]
    color = {"ok": (0, 200, 0), "marginal": (0, 180, 255), "small": (0, 0, 255)}.get(
        stats.verdict()[0], (160, 160, 160))

    # 参考框：手大致要撑到这个大小才谈得上判定
    cv2.rectangle(frame, (w // 2 - int(HAND_W_OK / 2), h // 2 - int(HAND_W_OK / 2)),
                  (w // 2 + int(HAND_W_OK / 2), h // 2 + int(HAND_W_OK / 2)),
                  (90, 90, 90), 1)
    cv2.putText(frame, f"target hand width >= {HAND_W_OK:.0f}px",
                (w // 2 - int(HAND_W_OK / 2), h // 2 - int(HAND_W_OK / 2) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    if lm is not None:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 4 if i in (4, 8, 12, 16, 20) else 3,
                       (255, 255, 255), -1)

    jit = stats.jitter
    lines = [
        f"hand width : {stats.mean_w:6.0f} px",
        f"detect rate: {stats.rate * 100:6.1f} %",
        f"jitter     : {'n/a' if jit is None else f'{jit:6.1f} px'}",
        extra,
    ]
    for i, t in enumerate(lines):
        if t:
            cv2.putText(frame, t, (12, 26 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    kind = stats.verdict()[0]
    tag = {"ok": "OK - usable", "marginal": "MARGINAL - too small",
           "small": "TOO SMALL - unusable", "none": "no hand"}.get(kind, "")
    cv2.putText(frame, tag, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def analyse_frame(lm_hand, lm_pose, frame):
    """返回 (hand_landmarks or None, 说明文字)"""
    return lm_hand, ""


def run_live(cam_idx, with_pose, backend, snap_dir):
    import cv2
    import mediapipe as mp
    hand_path = pm.find_model(HAND_MODEL)
    pose_path = pm.find_model(POSE_MODEL) if with_pose else None
    hand = make_hand(hand_path)
    pose = make_pose(pose_path) if pose_path else None
    cap = pm.open_camera(cam_idx, backend)
    stats = HandStats()

    print(f"手部模型: {os.path.basename(hand_path)}")
    if with_pose:
        print(f"姿态模型: {os.path.basename(pose_path) if pose_path else '（没找到，跳过）'}")
    print("把摄像头对准孩子写字的手 —— 目标是让手撑满参考框。")
    print("按键：s 存图 / q 或 ESC 退出\n")

    tick = 0          # ⚠️ 全局单调递增，两个模型共用也要各自递增，不能重置
    n = 0
    t_hand = []
    t_pose = []
    while True:
        fr = pm.read_frame(cap)
        if fr is None:
            continue
        h, w = fr.shape[:2]
        rgb = fr[:, :, ::-1].copy()
        mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        tick += 33
        t0 = time.perf_counter()
        rh = hand.detect_for_video(mi, tick)
        t_hand.append((time.perf_counter() - t0) * 1000)

        extra = ""
        if pose is not None:
            tick += 33
            t0 = time.perf_counter()
            rp = pose.detect_for_video(mi, tick)
            t_pose.append((time.perf_counter() - t0) * 1000)
            if rp.pose_landmarks:
                # 用"肩膀宽度占画面多大"来直观说明：身体取景合适时，手一定很小
                k = rp.pose_landmarks[0]
                sw = abs(k[11].x - k[12].x) * w
                extra = f"body shoulder {sw:.0f}px  (whole-body view)"
            else:
                extra = "body: not detected"

        lm = rh.hand_landmarks[0] if rh.hand_landmarks else None
        stats.add(lm, w, h)
        vis = draw_overlay(fr.copy(), lm, stats, extra)
        cv2.imshow("hand view check  (s=save  q=quit)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            os.makedirs(snap_dir, exist_ok=True)
            p = os.path.join(snap_dir, time.strftime("hand_%Y%m%d-%H%M%S.jpg"))
            cv2.imwrite(p, vis)
            print(f"  已存 {p}")
        n += 1

    cap.release()
    cv2.destroyAllWindows()
    hand.close()
    if pose is not None:
        pose.close()
    return stats, t_hand, t_pose, n


def run_image(path, with_pose):
    import cv2
    import mediapipe as mp
    fr = cv2.imread(path)
    if fr is None:
        sys.exit(f"读不了这张图：{path}")
    h, w = fr.shape[:2]
    hand = make_hand(pm.find_model(HAND_MODEL))
    pose = make_pose(pm.find_model(POSE_MODEL)) if with_pose else None
    rgb = fr[:, :, ::-1].copy()
    mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    t0 = time.perf_counter()
    rh = hand.detect_for_video(mi, 33)
    ms = (time.perf_counter() - t0) * 1000
    lm = rh.hand_landmarks[0] if rh.hand_landmarks else None
    stats = HandStats()
    stats.add(lm, w, h)

    extra = ""
    if pose is not None:
        rp = pose.detect_for_video(mi, 66)
        if rp.pose_landmarks:
            k = rp.pose_landmarks[0]
            extra = f"body shoulder {abs(k[11].x - k[12].x) * w:.0f}px"
    vis = draw_overlay(fr, lm, stats, extra)
    out = os.path.splitext(path)[0] + "_checked.jpg"
    cv2.imwrite(out, vis)
    hand.close()
    if pose is not None:
        pose.close()

    kind, msg = stats.verdict()
    print(f"图片尺寸：{w}x{h}    手部推理：{ms:.0f}ms")
    print(f"检测到手：{'是' if lm is not None else '否'}")
    print(f"结论：{msg}")
    print(f"标注图已存：{out}")
    return kind, msg


def selftest():
    print("=== 自检（不需要摄像头）===")
    p = pm.find_model(HAND_MODEL)
    print(f"手部模型：{p}")
    if not os.path.isfile(p):
        print(f"❌ 找不到 {HAND_MODEL}，请放到：")
        for d in pm.MODEL_DIRS:
            print(f"   {os.path.normpath(d)}")
        return 1
    print(f"  大小 {os.path.getsize(p) / 1024 / 1024:.1f} MB")
    hand = make_hand(p)
    import mediapipe as mp
    tick, ts = 0, []
    for _ in range(15):
        tick += 33
        img = np.full((720, 1280, 3), 200, dtype=np.uint8)
        t0 = time.perf_counter()
        hand.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), tick)
        ts.append((time.perf_counter() - t0) * 1000)
    hand.close()
    print(f"  空图 1280x720 推理均值 {sum(ts) / len(ts):.1f}ms（成本下限）")
    print("  统计类：", HandStats().verdict()[1])
    print("✅ 自检通过（模型可用、单帧成本可接受）")
    print("\n下一步：拿一张真实握笔照片跑 --image，先看手有多大。")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="握笔视角验证：先确认看得见，再谈判得准")
    ap.add_argument("--camera", type=int, default=None, help="摄像头索引")
    ap.add_argument("--image", default=None, help="用一张已有照片验证")
    ap.add_argument("--with-pose", action="store_true",
                    help="同时显示身体姿态（直观看到「一个摄像头兼顾不了」）")
    ap.add_argument("--backend", default=None, help="强制指定 cv2 后端，如 dshow")
    ap.add_argument("--snap-dir", default=os.path.join(HERE, "..", "hand_view"),
                    help="按 s 时存图的目录")
    ap.add_argument("--selftest", action="store_true", help="只验证模型与代码")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.image:
        run_image(args.image, args.with_pose)
        return 0
    if args.camera is None:
        ap.error("要么给 --camera N，要么给 --image 路径，要么 --selftest")

    stats, t_hand, t_pose, n = run_live(args.camera, args.with_pose,
                                        args.backend, args.snap_dir)
    kind, msg = stats.verdict()
    print("\n=== 小结 ===")
    print(f"跑了 {n} 帧")
    print(f"手部检出率：{stats.rate * 100:.1f}%   （长时间偏低说明机位/光照有问题）")
    print(f"手部平均宽度：{stats.mean_w:.0f} px")
    jit = stats.jitter
    print(f"跟踪抖动：{'样本不足' if jit is None else f'{jit:.1f} px'}")
    if t_hand:
        print(f"手部推理：{sum(t_hand) / len(t_hand):.1f}ms/帧")
    if t_pose:
        print(f"姿态推理：{sum(t_pose) / len(t_pose):.1f}ms/帧")
    print(f"\n结论：{msg}")

    if kind == "ok":
        print("→ 尺寸这一关过了。**但尺寸只是必要条件，不充分**：")
        print("  文献实测 MediaPipe 手部关节角误差约 22.5°，而握笔姿势的差异往往只有 10~30°，")
        print("  所以只能做**粗判**（拳握 / 三指大形态），别指望分清拇指搭在食指哪一节。")
    elif kind == "none":
        print("→ 完全没检测到手。先解决「手能不能进画面」，别的都别提。")
    else:
        print("→ 先把机位拉近/抬高，让手在画面里撑到参考框大小，再回来看结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

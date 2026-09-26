#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐姿判定逻辑 · 离线自检
=======================

**不需要摄像头、不需要 cv2 / mediapipe、不需要模型文件。**
用合成关键点直接回归 compute_* / judge_* / fuse，改判定逻辑后先跑这个。

    python tests/test_offline.py

每个用例是「构造关键点 -> 期望判定结果」，覆盖四类：
  A. 正面指标与判定
  B. 侧面指标与判定（含宽高比校正、偏航角补偿）
  C. 误差帧筛除（检测器误检不能污染基线）
  D. 双机位融合

坐标约定：MediaPipe 归一化，x∈[0,1] 向右，y∈[0,1] 向下（y 越大越靠画面下方）。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import posture_monitor as pm   # noqa: E402
import posture_engine as pe_mod   # noqa: E402

W16, H16 = 1280, 720   # 16:9
W43, H43 = 640, 480    # 4:3

FAILS = []
COUNT = 0


class KP:
    """一个关键点（模拟 MediaPipe NormalizedLandmark）。"""

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def kps(overrides):
    """33 个关键点，默认全在画面中心；overrides = {索引: (x, y) 或 (x, y, vis)}"""
    out = [KP() for _ in range(33)]
    for i, v in overrides.items():
        out[i] = KP(*v) if len(v) == 3 else KP(v[0], v[1])
    return out


# ---------------------------------------------------------------- 构造器

def front(span=0.20, ear_y=0.30, sh_y=0.60, ear_dy=0.0, sh_dy=0.0, cx=0.50):
    """正面机位：双肩 + 双耳。

    head_drop = (肩均y - 耳均y) / span，标准坐姿取 span=0.20 / ear_y=0.30 / sh_y=0.60
    -> head_drop = 1.5（落在合理区间 0.8~2.0 内）。
    """
    ls, rs = cx - span / 2.0, cx + span / 2.0
    return kps({
        pm.L_EAR: (cx - 0.06, ear_y),
        pm.R_EAR: (cx + 0.06, ear_y + ear_dy),
        pm.L_SH: (ls, sh_y),
        pm.R_SH: (rs, sh_y + sh_dy),
    })


def front_head_drop_ratio(span=0.20, ratio=0.0, sh_y=0.60):
    """构造「头部下沉比例为 ratio」的正面帧（基线的 head_drop = 1.5）。"""
    base_hd = (sh_y - 0.30) / span
    return front(span=span, ear_y=sh_y - span * base_hd * (1.0 - ratio), sh_y=sh_y)


def side(ear, sh, hip=None, hip_vis=1.0, ear_vis=1.0, sh_vis=1.0):
    """侧面机位：只让左侧（7/11/23）可信，右侧给低可见度，逼 _pick_side 选左侧。"""
    ov = {
        pm.L_EAR: (ear[0], ear[1], ear_vis),
        pm.L_SH: (sh[0], sh[1], sh_vis),
        pm.R_EAR: (ear[0], ear[1], 0.1),
        pm.R_SH: (sh[0], sh[1], 0.1),
    }
    if hip is not None:
        ov[pm.L_HIP] = (hip[0], hip[1], hip_vis)
        ov[pm.R_HIP] = (hip[0], hip[1], 0.1)
    return kps(ov)


# 侧面标准坐姿（像素空间：肩→耳 竖直 216px，肩→髋 竖直 216px）
S_EAR0 = (0.50, 0.30)
S_SH0 = (0.50, 0.60)
S_HIP0 = (0.52, 0.90)


def side_tilt(deg, length_px=216.0, w=W16, h=H16, hip=None):
    """头部绕肩旋转 deg 度（0=竖直向上），返回 (ear, sh)。"""
    import math
    dx = length_px * math.sin(math.radians(deg))
    dy = length_px * math.cos(math.radians(deg))
    return (S_SH0[0] + dx / w, S_SH0[1] - dy / h), S_SH0


def side_collapse(frac, w=W16, h=H16):
    """肩-耳距离相对基线收缩 frac 比例（趴桌 / 头下探），返回 (ear, sh)。"""
    length = 216.0 * (1.0 - frac)
    return (S_SH0[0], S_SH0[1] - length / h), S_SH0


def side_trunk(deg, w=W16, h=H16):
    """躯干前倾 deg 度，返回 hip 坐标。"""
    import math
    dx = 216.0 * math.tan(math.radians(deg))
    return (S_SH0[0] + dx / w, S_HIP0[1])


# ---------------------------------------------------------------- 用例框架

def case(name, got, want):
    global COUNT
    COUNT += 1
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}\n        期望 = {want!r}\n        实际 = {got!r}")
        FAILS.append(name)


def close(name, got, want, tol=1e-6):
    global COUNT
    COUNT += 1
    ok = got is not None and abs(got - want) <= tol
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}\n        期望 ≈ {want} (±{tol})\n        实际 = {got!r}")
        FAILS.append(name)


# ---------------------------------------------------------------- A. 正面

def test_front_metrics():
    print("\n[A] 正面指标与判定")

    base = pm.compute_metrics(front(), W16, H16)
    case("A1 标准坐姿帧不被筛除", base is not None, True)
    close("A2 head_drop = 1.5", base["head_drop"], 1.5)
    close("A3 head_tilt = 0", base["head_tilt"], 0.0)
    close("A4 span = 0.20", base["span"], 0.20)

    # 标定基线 + 各类坐姿
    baseline = {k: base[k] for k in ("head_drop", "head_tilt", "shoulder", "span")}

    def j(kp):
        return pm.judge(pm.compute_metrics(kp, W16, H16), baseline)

    case("A5 正常坐姿 -> 无问题", j(front()), [])
    case("A6 头部下沉 5% -> 不误报（迟滞靠时间，阈值内不该叫）",
         j(front_head_drop_ratio(ratio=0.05)), [])
    case("A7 头部下沉 12% -> 低头", j(front_head_drop_ratio(ratio=0.12)), ["低头"])
    case("A8 头部下沉 38% -> 低头", j(front_head_drop_ratio(ratio=0.38)), ["低头"])
    case("A9 歪头（双耳 7% 肩宽高差）-> 歪头",
         j(front(ear_dy=0.07 * 0.20)), ["歪头"])
    case("A10 歪肩（双肩 6% 肩宽高差）-> 歪肩",
         j(front(sh_dy=0.06 * 0.20)), ["歪肩"])

    # 距离过近：整体等比放大。肩宽 +20%，同时头肩距按同比例放大，
    # 于是 head_drop 不变 —— 这正是「按肩宽归一化」要达成的抗远近效果。
    near = front(span=0.24, ear_y=0.60 - (0.24 * 1.5))
    case("A11 靠近摄像头（等比放大 20%）-> 只报太近，不误报低头",
         j(near), ["太近"])

    # 正面指标按肩宽归一化，宽高比影响在「当前 / 基线」处抵消
    m169 = pm.compute_metrics(front(), W16, H16)
    m43 = pm.compute_metrics(front(), W43, H43)
    case("A12 同一归一化帧在 16:9 与 4:3 下正面指标一致（肩宽归一化天然抗宽高比）",
         round(m169["head_drop"], 9) == round(m43["head_drop"], 9), True)


# ---------------------------------------------------------------- B. 侧面

def test_side_metrics():
    print("\n[B] 侧面指标与判定")

    base = pm.compute_side_metrics(side(S_EAR0, S_SH0, S_HIP0), W16, H16)
    case("B1 标准侧面帧不被筛除", base is not None, True)
    close("B2 neck_angle = 0", base["neck_angle"], 0.0, 1e-6)
    close("B3 sh_ear_len = 216px", base["sh_ear_len"], 216.0, 1e-6)
    close("B4 trunk_angle = atan(25.6/216) ≈ 6.76°",
          base["trunk_angle"], 6.7601, 0.01)

    baseline = dict(base)

    def j(kp, w=W16, h=H16, b=None):
        m = pm.compute_side_metrics(kp, w, h)
        if m is None:
            return None
        return pm.judge_side(m, b or baseline)

    case("B5 标准坐姿 -> 无问题", j(side(S_EAR0, S_SH0, S_HIP0)), [])

    ear, sh = side_tilt(15.0)
    case("B6 头前倾 15° -> 头前倾", j(side(ear, sh, S_HIP0)), ["头前倾"])

    ear, sh = side_tilt(8.0)
    case("B7 头前倾 8°（低于 12° 阈值）-> 不误报", j(side(ear, sh, S_HIP0)), [])

    case("B8 塌腰 +10° -> 塌腰",
         j(side(S_EAR0, S_SH0, side_trunk(16.76))), ["塌腰"])

    case("B9 髋部被桌面遮挡 -> 跳过躯干项，不误报",
         j(side(S_EAR0, S_SH0, S_HIP0, hip_vis=0.2)), [])

    case("B10 髋不可见 + 头前倾 -> 只报头前倾",
         j(side(*side_tilt(15.0), S_HIP0, hip_vis=0.2)), ["头前倾"])

    ear, sh = side_collapse(0.20)
    case("B11 肩-耳距离收缩 20% -> 低头（趴桌/头下探）", j(side(ear, sh, S_HIP0)), ["低头"])

    ear, sh = side_collapse(0.06)
    case("B12 收缩 6%（低于 12% 阈值）-> 不误报", j(side(ear, sh, S_HIP0)), [])

    # 宽高比校正 —— 这是本次修复的核心，旧实现会把角度压扁约 44%
    ear = (0.55, 0.30)
    sh = (0.50, 0.60)
    m169 = pm.compute_side_metrics(side(ear, sh), W16, H16)
    m43 = pm.compute_side_metrics(side(ear, sh), W43, H43)
    import math
    want169 = math.degrees(math.atan2(0.05 * W16, 0.30 * H16))   # 16.50°
    want43 = math.degrees(math.atan2(0.05 * W43, 0.30 * H43))    # 12.53°
    naive = math.degrees(math.atan2(0.05, 0.30))                 # 9.46°（旧算法的错误值）
    close("B13 16:9 下角度 = 像素空间真实角 16.50°", m169["neck_angle"], want169, 1e-6)
    close("B14 4:3 下角度 = 像素空间真实角 12.53°", m43["neck_angle"], want43, 1e-6)
    case("B15 新旧算法确实不同（旧实现会算成 9.46°）",
         abs(m169["neck_angle"] - naive) > 5.0, True)

    # 偏航角补偿
    pm.SIDE_YAW_DEG = 45.0
    m45 = pm.compute_side_metrics(side(ear, sh), W16, H16)
    want45 = math.degrees(math.atan2(0.05 * W16 * math.sqrt(2), 0.30 * H16))
    close("B16 偏航 45° 下按 1/sin(yaw) 还原到矢状面 ≈ 22.74°",
          m45["neck_angle"], want45, 1e-6)
    case("B17 偏航 45° 测得角 > 正侧 90°（说明斜侧机会压低读数）",
         m45["neck_angle"] > m169["neck_angle"], True)
    pm.SIDE_YAW_DEG = 90.0

    # 筛除超限帧
    case("B18 头低到肩以下（neck_angle 超 60°）-> 丢弃该帧",
         pm.compute_side_metrics(
             side((0.99, 0.62), S_SH0), W16, H16), None)
    case("B19 肩-耳距离过小（关键点挤成一团）-> 丢弃该帧",
         pm.compute_side_metrics(
             side((0.50, 0.599), S_SH0), W16, H16), None)


# ---------------------------------------------------------------- C. 误差帧筛除

def test_screen():
    print("\n[C] 误检帧筛除（防污染基线）")

    case("C1 肩宽仅占画面 3.7% -> 丢弃（实测误检帧的典型特征）",
         pm.compute_metrics(front(span=0.037), W16, H16), None)
    case("C2 肩宽占满 98% -> 丢弃",
         pm.compute_metrics(front(span=0.98), W16, H16), None)
    case("C3 耳肩距仅 0.2 倍肩宽 -> 丢弃",
         pm.compute_metrics(front(span=0.20, ear_y=0.60 - 0.20 * 0.2), W16, H16), None)
    case("C4 耳肩距达 5 倍肩宽 -> 丢弃",
         pm.compute_metrics(front(span=0.20, ear_y=0.60 - 0.20 * 5.0), W16, H16), None)
    case("C5 画面里没有人（关键点全在中心，肩宽=0）-> 丢弃",
         pm.compute_metrics(kps({}), W16, H16), None)

    # build_baseline 要容忍各样本键不一致（侧面髋部时有时无）
    b = pm.build_baseline([
        {"neck_angle": 1.0, "trunk_angle": 5.0},
        {"neck_angle": 2.0},
        {"neck_angle": 3.0, "trunk_angle": 7.0},
    ])
    case("C6 build_baseline 容忍键不一致（缺 trunk_angle 的样本不参与该项）",
         (b["neck_angle"], b["trunk_angle"]), (2.0, 6.0))
    case("C7 中位数抗单帧抖动（[1,2,3] -> 2.0 而非均值）", b["neck_angle"], 2.0)


# ---------------------------------------------------------------- D. 融合

def test_fuse():
    print("\n[D] 双机位融合")

    case("D1 正面低头 + 侧面前倾 -> 升级判为含胸驼背",
         pm.fuse(["低头"], ["头前倾"]), ["含胸驼背"])
    case("D2 只有侧面头前倾 -> 不升级，保留单项",
         pm.fuse([], ["头前倾"]), ["头前倾"])
    case("D3 只有正面低头 -> 不升级，保留单项",
         pm.fuse(["低头"], []), ["低头"])
    case("D4 多问题并存：升级项排首位，其余保持顺序",
         pm.fuse(["低头", "歪头"], ["头前倾", "塌腰"]),
         ["含胸驼背", "歪头", "塌腰"])
    case("D5 正面塌腰 + 侧面塌腰 -> 不合并（不是互证对）",
         pm.fuse(["歪肩"], ["塌腰"]), ["歪肩", "塌腰"])


# ---------------------------------------------------------------- E. 迟滞

def test_gate():
    print("\n[E] 迟滞与提醒间隔（ReminderGate，CLI 与 GUI 共用）")

    g = pm.ReminderGate(sustain=4.0, clear=3.0, gap=90.0)

    st = g.update(["头前倾"], 0.0)
    case("E1 单帧越界 -> 不确认（抖一下绝不该叫）", (st.confirmed, st.fire), (False, None))

    st = g.update(["头前倾"], 3.9)
    case("E2 连续 3.9s（差 0.1s 到门槛）-> 仍未确认", (st.confirmed, st.fire), (False, None))

    st = g.update(["头前倾"], 4.0)
    case("E3 连续满 4.0s -> 确认并播报一次", (st.confirmed, st.fire), (True, "头前倾"))

    st = g.update(["头前倾"], 30.0)
    case("E4 持续超标但未到 90s 间隔 -> 不重复播报", (st.confirmed, st.fire), (True, None))

    st = g.update(["头前倾"], 95.0)
    case("E5 距上次播报 91s -> 再次播报", st.fire, "头前倾")

    st = g.update([], 96.0)
    case("E6 刚恢复 1s（未到 3s）-> 尚未解除", (st.issue, st.cleared), ("头前倾", False))

    st = g.update([], 99.0)
    case("E7 恢复满 3s -> 解除并标记 cleared（可做「坐正了真棒」正反馈）",
         (st.issue, st.cleared), (None, True))

    g2 = pm.ReminderGate(4.0, 3.0, 90.0)
    g2.update(["头前倾"], 0.0)
    g2.update(["头前倾"], 3.5)
    st = g2.update(["塌腰"], 3.6)
    case("E8 问题切换 -> 观察窗口重置，不继承上一个问题的超标时长",
         (st.issue, st.confirmed), ("塌腰", False))
    st = g2.update(["塌腰"], 7.8)
    case("E9 切换后重新累计满 4s -> 播报新问题", st.fire, "塌腰")

    # 恢复期内界面应继续显示原问题，否则会闪（E6 就是这条的回归锁）
    g4 = pm.ReminderGate(4.0, 3.0, 90.0)
    g4.update(["塌腰"], 0.0)
    g4.update(["塌腰"], 4.0)
    st = g4.update([], 4.2)
    case("E12 恢复期第 1 帧 -> 仍报原问题，界面不闪", st.issue, "塌腰")
    st = g4.update([], 7.3)
    case("E13 恢复满 3s -> 才转为 None", (st.issue, st.cleared), (None, True))

    g3 = pm.ReminderGate(4.0, 3.0, 90.0)
    last = None
    for t in range(60):
        last = g3.update([], float(t))
    case("E10 一直良好 -> 永不报警", (last.issue, last.fire, last.cleared), (None, None, False))

    g3.reset()
    case("E11 reset() 清空状态（重新标定/换机位后必须调用）",
         (g3.issue, g3.good_since, len(g3.last_fired)), (None, None, 0))

    # ---- 任意两次提醒之间的全局下限（跨问题也生效）----
    # 复现实测 bug：只按问题分别计时的话，头前倾/塌腰/低头 各自都没到间隔，
    # 却会轮流触发 —— 变成每几秒响一次。
    # 需求原话：「三十秒之内如果提示过不正确就不提示，即便不正确的原因也不一样。」
    g5 = pm.ReminderGate(sustain=4.0, clear=3.0, gap=90.0, any_gap=30.0)
    st = g5.update(["头前倾"], 0.0)
    st = g5.update(["头前倾"], 4.0)
    case("E14 首次提醒正常触发", st.fire, "头前倾")
    g5.update(["塌腰"], 5.0)
    st = g5.update(["塌腰"], 9.0)
    case("E15 **换了原因**、仅隔 5 秒 -> 照样被拦（这正是需求要的行为）",
         st.fire, None)
    st = g5.update(["塌腰"], 20.0)
    case("E16 距上次提醒 16s（<30s）-> 仍不提醒", st.fire, None)
    st = g5.update(["塌腰"], 34.1)
    case("E17 距上次提醒 30.1s -> 才允许下一次", st.fire, "塌腰")
    case("E18 默认全局下限 = 30 秒（需求指定的值，别被改回大或小）",
         pm.REMIND_GAP_ANY_SEC, 30.0)
    case("E19 默认参数确实用上了这个值",
         pm.ReminderGate().any_gap, 30.0)

    # ---- 播放队列：满了要顶掉旧的，不能阻塞 ----
    # 注意：先 stop() 再等线程退出，这样只验队列语义、不会真的出声。
    sp = pe_mod.Speaker(voice_dir=pm.VOICE_DIR, mute=True)
    sp.stop()
    time.sleep(0.35)
    case("E20 静音时 say() 返回 False 且不入队", (sp.say("塌腰"), sp._q.qsize()), (False, 0))
    sp.mute = False
    case("E21 say() 立即返回布尔值（不阻塞调用方）",
         isinstance(sp.say("塌腰"), bool), True)
    sp.say("头前倾")
    case("E22 队列深度为 1（提醒有时效性，不该排队积压）", sp._q.maxsize, 1)
    case("E23 队满时新条目顶掉旧的，队列始终只留 1 条", sp._q.qsize(), 1)
    kind, arg = sp._q.get_nowait()
    case("E24 留在队里的是最新那条（旧的被丢弃）",
         (kind, "头前倾" in arg), ("file", True))
    case("E25 语音可枚举（含中文文件名）",
         "头前倾" in pe_mod.Speaker.available(), True)


# ---------------------------------------------------------------- H. 静音约束

def test_silent_when_good():
    """坐正了**不许发声**。

    这条是需求约束（「坐姿没问题就不要发出任何声音」），但它落在引擎的
    采集循环里 —— 没有摄像头就没法跑。所以用**源码级回归锁**代替：
    锁住"不再出现这个调用"，改动时立刻失败。
    """
    print("\n[H] 坐正了不发声（需求约束，源码级回归锁）")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "posture_engine.py"), encoding="utf-8").read()
    case("H1 引擎里已无「坐正->提示音」这个调用",
         "_speaker.chime(\"good\")" in src, False)
    case("H2 标定完成的提示音仍保留（那是系统事件，不是坐姿判定）",
         "_speaker.chime(\"done\")" in src, True)
    app_src = open(os.path.join(here, "posture_app.py"), encoding="utf-8").read()
    case("H3 演示模式同样只在 warn 时发声",
         "_demo_speaker.chime(\"good\")" in app_src, False)


# ---------------------------------------------------------------- F. 标定自检

def test_sanity():
    print("\n[F] 标定结果自检（GUI 靠它上色）")

    rows = pm.sanity_rows(["侧面"], [{"neck_angle": 8.0, "trunk_angle": 6.0}])
    case("F1 全部合理 -> 无 warn", [r[2] for r in rows], ["ok", "ok"])

    rows = pm.sanity_rows(["侧面"], [{"neck_angle": 35.0}])
    case("F2 neck_angle 超 20° -> 标 warn", rows[0][2], "warn")

    rows = pm.sanity_rows(["侧面"], [{"neck_angle": 5.0}])
    case("F3 髋部拿不到 -> 标 info 而非 warn（属正常，不是故障）",
         rows[1][2], "info")

    rows = pm.sanity_rows(["正面"],
                          [{"span": 0.03, "head_drop": 1.5, "head_tilt": 0.0, "shoulder": 0.0}])
    case("F4 肩宽只占 3% -> 标 warn（典型误检基线）", rows[0][2], "warn")


# ---------------------------------------------------------------- G. 阈值建议

def test_threshold_suggest():
    print("\n[G] 阈值建议（人工审核数据 -> 建议阈值）")

    r = pm.suggest_threshold([20.0, 22.0], [3.0, 5.0], current=12.0)
    case("G1 完美可分 -> 给出建议值", r is not None, True)
    case("G2 建议值落在两类之间（5~20）", 5.0 <= r["suggested"] <= 20.0, True)
    case("G3 建议阈值下误报与漏报都为 0",
         (r["fp_at_suggested"], r["fn_at_suggested"]), (0, 0))
    case("G4 当前阈值 12° 下的误报数", r["fp_at_current"], 0)

    r = pm.suggest_threshold([10.0, 14.0], [8.0, 12.0], current=12.0)
    case("G5 两类重叠 -> 仍给建议值（不硬猜）", r is not None, True)
    case("G6 重叠时优先少误报：建议阈值下误报为 0", r["fp_at_suggested"], 0)

    r = pm.suggest_threshold([10.0], [10.0], current=12.0)
    case("G7 完全同分 -> 误报权重(1.5)生效，选不误报的那一侧",
         r["fp_at_suggested"], 0)

    case("G8 只有一类样本 -> 不给建议（避免瞎调）",
         pm.suggest_threshold([], [1.0]), None)
    case("G9 全是 None -> 不给建议",
         pm.suggest_threshold([None], [None]), None)

    # side_score：方向必须统一成「越大越该报警」，否则扫阈值会反
    m = {"neck_angle": 20.0, "sh_ear_len": 180.0}
    b = {"neck_angle": 5.0, "sh_ear_len": 200.0}
    close("G10 neck_angle 分数 = 当前 - 基线", pm.side_score("neck_angle", m, b), 15.0)
    close("G11 head_collapse 分数 = 收缩比例（方向与角度指标相反）",
          pm.side_score("head_collapse", m, b), 0.10, 1e-9)
    case("G12 髋部不可见（无 trunk_angle）-> 返回 None",
         pm.side_score("trunk_angle", m, b), None)
    case("G13 未知指标 -> None", pm.side_score("whatever", m, b), None)


# ---------------------------------------------------------------- 入口

def main():
    print("=" * 62)
    print("坐姿判定逻辑 · 离线自检（合成关键点，无需摄像头/依赖）")
    print("=" * 62)
    test_front_metrics()
    test_side_metrics()
    test_screen()
    test_fuse()
    test_gate()
    test_sanity()
    test_threshold_suggest()
    test_silent_when_good()

    print("\n" + "=" * 62)
    if FAILS:
        print(f"结果：{COUNT - len(FAILS)}/{COUNT} 通过，{len(FAILS)} 个失败")
        for f in FAILS:
            print(f"  ✗ {f}")
        sys.exit(1)
    print(f"结果：{COUNT}/{COUNT} 全部通过 ✅")


if __name__ == "__main__":
    main()

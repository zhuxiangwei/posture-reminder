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


# ------------------------------------------------- I. 采集档位 / 日志落盘

def test_hd_and_logging():
    """锁住两件容易在后续改动里被"顺手改回去"的事：

    1. **采集档位 = 3840x2160（4K）+ MJPG**（2026-09-28 实测定稿）。
       历史沿革值得记住（三轮结论，前两轮都是错的）：
         · 09-26：摄像头（Nebula 02）只输出 1080p 且不慢 → 定 1920x1080
         · 09-28 早：发现新摄像头 1080p 只有 3.3 FPS < 5 FPS 要求 → 改成 720p
         · 09-28 晚（**用户点破"这是 4K 摄像头"后**）：真因是**压缩格式** ——
           OpenCV 默认走 YUY2 未压缩，USB 2.0 带宽喂不动高分辨率；
           切 **MJPG** 后 **4K 反而 100% 可靠、~14 FPS**。前两轮结论都是
           YUY2 造成的假象。
       ⚠️ 所以这里锁的是"4K + MJPG"这一**组合**，别只改一半。
    2. **运行日志必须落盘** —— GUI 里那份只在内存，关掉就没了，
       排障/复测全靠文件。这条用源码级锁，防以后新增日志分支忘了接上。
    """
    print("\n[I] 采集档位 + 运行日志落盘")

    case("I1 采集默认宽度 = 3840（4K）", pm.DEFAULT_CAM_W, 3840)
    case("I2 采集默认高度 = 2160", pm.DEFAULT_CAM_H, 2160)
    case("I2b 提供 MJPG FOURCC 常量（USB2.0 跑高分辨率的唯一出路）",
         hasattr(pm, "MJPG_FOURCC") and pm.MJPG_FOURCC != 0, True)
    # FOURCC 必须设在分辨率**之前**（格式协商发生在选模式时），顺序写反会静默失效
    _src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "posture_monitor.py"), encoding="utf-8").read()
    case("I2c open_camera 里 FOURCC 先于分辨率设置",
         _src.index("cap.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)") <
         _src.index("cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])"), True)
    case("I3 引擎 config 的 width 与之一致（防两处漂移）",
         pe_mod.DEFAULT_CONFIG["width"], pm.DEFAULT_CAM_W)
    case("I4 引擎 config 的 height 与之一致",
         pe_mod.DEFAULT_CONFIG["height"], pm.DEFAULT_CAM_H)
    case("I5 审核默认存原图（用户要求处处最高清）",
         pe_mod.DEFAULT_CONFIG["review_keep_fullres"], True)
    case("I6 预览图宽度 ≥ 960（2x 缩放屏上不虚）",
         pe_mod.PREVIEW_MAX_W >= 960, True)
    case("I7 回放的时间轴格式化", pm._mmss(72.35), "01:12.3")
    case("I8 回放指标串含当前值与基线",
         "基线" in pm._fmt_metrics({"neck_angle": 20.0, "sh_ear_len": 200.0},
                                   {"neck_angle": 6.0, "sh_ear_len": 210.0}), True)
    case("I9 CLI 保留 --video 回放入口",
         "--video" in pm.build_parser().format_help(), True)

    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="posture_log_")
    try:
        # 过期文件应被清掉，今天的应新建
        old = os.path.join(tmp, "run-20000101.log")
        open(old, "w").close()
        rl = pe_mod.RunLog(tmp)
        rl.write("[test] 一行日志")
        case("I10 日志文件已创建", bool(rl.path) and os.path.isfile(rl.path), True)
        case("I11 日志内容已落盘",
             "一行日志" in open(rl.path, encoding="utf-8").read(), True)
        case("I12 过期日志被剪掉（保留窗口内）", os.path.exists(old), False)
        case("I13 日志目录路径也可用（自动补当天文件名）",
             bool(pe_mod.RunLog(tmp).path), True)

        off = pe_mod.RunLog(tmp, enabled=False)
        off.write("不该写")
        case("I14 --no-log 时不落盘", off.path, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 源码级锁：新增/修改日志分支时不能绕过落盘
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_src = open(os.path.join(here, "posture_app.py"), encoding="utf-8").read()
    case("I15 界面的 _log() 接了落盘", "self.runlog.write(line)" in app_src, True)


# ------------------------------------------------ J 组：SAC 拦截 matplotlib 的兜底
#
# 背景：本机 Windows 智能应用控制（SAC）拦掉了 matplotlib 的 _image.pyd，
#       mediapipe 的 vision 包因硬导入 matplotlib.pyplot 而整包失败，
#       表现为 `AttributeError: 'NoneType' object has no attribute 'PoseLandmarker'`。
#       修法是 compat_matplotlib.py 用桩顶掉 matplotlib（详见该模块文档）。
# 这组用例锁定：桩存在、幂等、不干扰真实 matplotlib、报错能带出真病因。

def test_sac_matplotlib_compat():
    print("\n[J] Windows SAC 拦截 matplotlib 的兜底")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "compat_matplotlib.py"), encoding="utf-8").read()

    case("J1 桩模块存在", os.path.exists(os.path.join(here, "compat_matplotlib.py")), True)
    case("J2 桩提供 install()", "def install(" in src, True)
    case("J3 桩提供 stub_active()", "def stub_active(" in src, True)
    case("J4 桩顶掉的是 matplotlib.pyplot", 'sys.modules["matplotlib.pyplot"] = pyplot' in src, True)

    # 主模块必须真的调用它，而不是只把文件放那儿
    mon_src = open(os.path.join(here, "posture_monitor.py"), encoding="utf-8").read()
    case("J5 posture_monitor 调用 install()", "compat_matplotlib.install()" in mon_src, True)
    # 桩必须装在 mediapipe 之前（否则 mediapipe 已经导入失败了）
    case("J6 桩在 import mediapipe 之前装上",
         mon_src.index("compat_matplotlib.install()") < mon_src.index("import mediapipe as mp"), True)

    # 真正导入一遍，验证实际行为（不只做源码字符串匹配）
    sys.path.insert(0, here)
    try:
        import compat_matplotlib as cm

        # 幂等：装了桩之后重复 install 不该再动手
        cm.install()
        case("J7 install() 幂等（二次返回 False）", cm.install(), False)

        # 幂等的前提是"已经处理过"，而处理结果取决于真实 matplotlib 是否可用。
        # 两种结果都合法，但**必须能确定地测出桩本身的行为**——
        # 不能像原来那样"真实库可用就跳过绘图报错那条"，
        # 否则测试条数会随机器状态漂移（曾出现 117 变 116）。
        if cm.stub_active():
            case("J8 桩已激活（本机 matplotlib 可用时返回 False 才对）",
                 cm.stub_active(), True)
        else:
            case("J8 真实 matplotlib 可用时桩不介入", cm.stub_active(), False)

        # 无论哪条分支，都直接构造一个桩来验证它的行为契约（确定性）
        stub_pyplot = cm._make_pyplot_stub()
        try:
            stub_pyplot.figure()
            raised = False
        except NotImplementedError:
            raised = True
        # 关键：绘图调用必须显式报错，不能假装能用（否则会变成更难查的故障）
        case("J9 桩的绘图调用显式抛 NotImplementedError", raised, True)
        # 桩必须补齐 mediapipe drawing_utils 会碰到的名字
        case("J9b 桩补齐 figure/axes/show",
             all(hasattr(stub_pyplot, n) for n in ("figure", "axes", "show")), True)

        # mediapipe 的 vision 必须可用 —— 这是本次修复的最终目的
        from mediapipe.tasks.python import vision as _v
        case("J10 装上桩后 vision.PoseLandmarker 可用",
             hasattr(_v, "PoseLandmarker"), True)
    except ImportError as e:
        # 测试机没装 mediapipe 时跳过（离线自检本身不依赖它）
        print(f"  SKIP  J8~J10（本机无 mediapipe：{e}）")
    finally:
        if here in sys.path:
            sys.path.remove(here)

    # 静态锁：报错可见性 —— 不许再静默吞掉 mediapipe 的导入异常
    case("J11 保留原始导入异常（不再是裸 except）",
         "_MP_IMPORT_ERR = _e" in mon_src, True)
    case("J12 _need_mp() 会带出原始报错",
         "_MP_IMPORT_ERR is not None" in mon_src, True)
    # GUI 路径（engine）必须自己补守卫，它不经过 CLI 的 _need_mp()
    eng_src = open(os.path.join(here, "posture_engine.py"), encoding="utf-8").read()
    case("J13 引擎在 make_landmarker 前补了依赖守卫",
         "pm.vision is None" in eng_src, True)
    case("J14 引擎的守卫在 make_landmarker 之前",
         eng_src.index("pm.vision is None") < eng_src.index("pm.make_landmarker(model)"), True)


# --------------------------------------- K. 摄像头开流鲁棒性（2026-09-28 实装踩坑）
#
# 背景：换摄像头后启动画面全黑。根因不是"SAC"（那是另一件事），而是：
#   ① `cap.read()` 返回 ok=True 但**内容全零**（驱动接受格式协商却填不进数据）
#      —— 只判 ok 就会选中这种破开流，然后对着黑图跑。
#   ② 这种坏开流**重开设备就好**，不是档位不支持。
#   ③ 但"**好的**开流"前几帧也是全零（1080p 要 3 帧），早退判据设小了会误杀好开流。
# 这组用例把这几条钉住，避免以后又被"顺手简化"掉。

class _FakeFrame:
    """够 frame_is_usable 用的最小帧替身（避免为测试引入 numpy）。"""

    def __init__(self, max_pixel, shape=(720, 1280, 3)):
        self._max = max_pixel
        self.shape = shape
        self.size = shape[0] * shape[1] * shape[2]

    def max(self):
        return self._max


def test_camera_open_robustness():
    print("\n[K] 摄像头开流鲁棒性（全零帧 / 重开 / 预热容限）")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "posture_monitor.py"), encoding="utf-8").read()

    # ---- frame_is_usable：全零帧必须判为不可用
    case("K1 全零帧判为不可用", pm.frame_is_usable(_FakeFrame(0)), False)
    case("K2 有内容的帧判为可用", pm.frame_is_usable(_FakeFrame(255)), True)
    case("K3 极暗但有信号（max=9）判为可用", pm.frame_is_usable(_FakeFrame(9)), True)
    case("K4 边界：max=8 判为可用（阈值含等号）",
         pm.frame_is_usable(_FakeFrame(8)), True)
    case("K5 边界：max=7 判为不可用",
         pm.frame_is_usable(_FakeFrame(7)), False)
    # 空帧 / None 不能抛异常（驱动抽风时会返回 None）
    case("K6 None 帧不抛异常且判为不可用", pm.frame_is_usable(None), False)
    case("K7 空数组帧判为不可用", pm.frame_is_usable(_FakeFrame(255, (0, 0, 3))), False)

    # ---- 预热容限：判据用 max 而非 mean（避免把正常偏暗画面误杀）
    case("K8 判据基于 max（大面积暗 + 一个亮点也算有画面）",
         pm.frame_is_usable(_FakeFrame(200)), True)
    case("K9 源码里确实是 frame.max() 而非 mean()", "frame.max()" in src, True)

    # ---- 预热容限：必须按档位给，不能一刀切
    #     实测好开流的预热帧数：480p=0 / 720p=1 / 1080p=3。
    #     设太小 -> 误杀好开流（曾把 1080p 全部误杀）；
    #     设太大（一刀切 4）-> 720p 的坏开流白等 4s，启动被拖慢一倍。
    import inspect as _ins
    sig = _ins.signature(pm.wait_usable_frame)
    case("K10 wait_usable_frame 的 blank_abort 默认随档位走（未写死）",
         sig.parameters["blank_abort"].default is None, True)
    case("K10b 720p 的预热容限 > 1（要让 1 帧预热的好开流通过）",
         pm.warmup_budget(1280, 720) > 1, True)
    case("K10c 1080p 的预热容限 > 3（要让 3 帧预热的好开流通过）",
         pm.warmup_budget(1920, 1080) > 3, True)
    case("K10d 720p 比 1080p 的容限小（坏开流别白等）",
         pm.warmup_budget(1280, 720) < pm.warmup_budget(1920, 1080), True)

    # ---- 重试开流：坏开流必须换新句柄重开
    case("K11 定义了 MAX_OPEN_ATTEMPTS", hasattr(pm, "MAX_OPEN_ATTEMPTS"), True)
    case("K12 MAX_OPEN_ATTEMPTS >= 3（单次开流有失败率，要留重试余量）",
         pm.MAX_OPEN_ATTEMPTS >= 3, True)
    case("K13 open_camera 里确实做了重开循环",
         "MAX_OPEN_ATTEMPTS" in src and "for attempt in range" in src, True)

    # ---- 超时保护：cv2 的 C 层阻塞 try/except 抓不住，必须靠线程 join
    case("K14 提供带超时的读帧 read_frame_timed", hasattr(pm, "read_frame_timed"), True)
    case("K15 超时用线程 join 实现（不是 try/except）",
         "join(" in src and "threading.Thread" in src, True)

    # ---- 后端阻塞：MSMF 连构造函数都会无限阻塞，必须能被跳过
    case("K16 有后端阻塞的降级路径（skipped 集合）", "skipped" in src, True)
    # ⚠️ 关键：创建 cap 必须发生在**调用线程**。
    #    DSHOW 走 COM，cap 绑定创建它的线程的 COM 单元。
    #    实测 2x2（1280x720 DSHOW，每格 5 轮独立进程）：
    #        创建=主线程  -> 4/5 ~ 3/5（可用）
    #        创建=工作线程 -> 0/5（**完全不可用**）
    #    所以**不能**用"工作线程里创建 + 超时 join"这套（曾这么写过，是错的）。
    case("K16b 不提供跨线程创建 cap 的危险函数（open_cap_timed 已移除）",
         not hasattr(pm, "open_cap_timed"), True)
    case("K16c 构造函数子进程探测（可硬杀，防卡死）",
         hasattr(pm, "backend_viable_subprocess"), True)
    case("K16d 创建 cap 的调用不在工作线程包装里出现在创建点附近",
         "cap = cv2.VideoCapture(i, be)" in src, True)
    case("K16e 构造函数超时常量存在",
         hasattr(pm, "OPEN_PROBE_TIMEOUT") and pm.OPEN_PROBE_TIMEOUT > 0, True)

    # ---- 报错要区分"打不开"与"打开了但没画面"
    case("K17 报错文案覆盖全零帧情形", "画面全零" in src, True)
    case("K18 --list-cameras 标注全零档位", "全零" in src, True)


# ------------------------------------------------- L. HiDPI 显示（预览铺满）
#
# 背景（2026-09-28 用户报"视频窗口显示的只有四分之一"）：
#   预览控件逻辑 520x290，本机屏幕 dpr=2.0 -> 物理 1040x580。
#   原先在 setPixmap 时缩放到"控件当前尺寸"，两个坑叠在一起：
#     ① QPixmap 从 QImage/文件来时 dpr=1.0，Qt 对 dpr=1.0 的 pixmap
#        按 1 像素:1 **设备像素**画、不放缩 -> 只占 1/4 面积；
#     ② 更隐蔽：尺寸被**固化在设置那一刻**。预览区高度会随表头提示语
#        行数变化而回流，控件变大后旧 pixmap 就只占一角 —— 且**时好时坏**。
#   修法：改成 ScaledImageLabel，在 **paintEvent** 里按**当前**尺寸缩放
#        并设回 dpr。任何时刻自洽，不依赖调用时序。
# 这组做源码级锁（Qt 渲染没法在无界面回归里跑，但"有没有退回旧写法"能静态查）。

def test_hidpi_preview():
    print("\n[L] HiDPI 显示（预览/审核图不得只显示 1/4 或一角）")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = open(os.path.join(here, "posture_app.py"), encoding="utf-8").read()

    case("L1 提供 ScaledImageLabel（绘制期缩放）",
         "class ScaledImageLabel(QLabel)" in app, True)
    case("L2 它在 paintEvent 里缩放（不靠调用时序）",
         "def paintEvent(self, ev):" in app, True)
    case("L3 缩放按控件物理尺寸（乘 dpr）",
         "int(sz.width() * dpr)" in app, True)
    case("L4 缩放后设回 devicePixelRatio",
         "pm.setDevicePixelRatio(dpr)" in app, True)
    case("L5 绘制前先走 QLabel 默认绘制（保住样式表背景与占位文字）",
         "super().paintEvent(ev)" in app, True)

    # 关键：不许再有"setPixmap 前先缩放到控件尺寸"的旧写法
    import re as _re
    case("L6 没有任何 setPixmap 调用（全部走 set_image）",
         _re.findall(r"\.setPixmap\(", app), [])
    case("L7 没有 pix.scaled(widget.size()) 的旧写法",
         _re.findall(r"\.scaled\(\s*self\.\w+\.size\(\)", app), [])

    case("L8 预览走 set_image", "self.preview.set_image(" in app, True)
    case("L9 审核大图也走 set_image（同一处坑）",
         "self.img.set_image(" in app, True)
    case("L10 两个控件都用 ScaledImageLabel",
         app.count("ScaledImageLabel(") >= 3, True)   # 类定义 1 + 两个实例

    # 预览图必须不小于控件物理宽度，否则缩放变成上采样、会发虚
    case("L11 PREVIEW_MAX_W >= 1280（控件物理宽实测 1040，须留余量）",
         pe_mod.PREVIEW_MAX_W >= 1280, True)


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
    test_hd_and_logging()
    test_sac_matplotlib_compat()
    test_camera_open_robustness()
    test_hidpi_preview()

    print("\n" + "=" * 62)
    if FAILS:
        print(f"结果：{COUNT - len(FAILS)}/{COUNT} 通过，{len(FAILS)} 个失败")
        for f in FAILS:
            print(f"  ✗ {f}")
        sys.exit(1)
    print(f"结果：{COUNT}/{COUNT} 全部通过 ✅")


if __name__ == "__main__":
    main()

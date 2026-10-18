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

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import posture_monitor as pm   # noqa: E402
import posture_engine as pe_mod   # noqa: E402

# ---------------------------------------------------------------- 源码级回归锁的"查哪个文件"
#
# 本文件里有几组用例是**静态查源码**的（Qt 渲染、开流重试、依赖守卫这类
# 没法在无摄像头/无界面环境里跑的行为）。重构后实现都搬进了 `posture/` 包，
# 所以这里集中把"文件路径"定义好 —— 以后搬家只改这一处，不许在每条 case 里现写路径。
# ⚠️ 路径写死成"包内模块"而不是兼容壳：壳里只有 import 语句，
#    查它等于什么都不查（这是最容易悄悄失效的一类测试）。

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(relpath):
    p = os.path.join(_HERE, relpath)
    with open(p, encoding="utf-8") as f:
        return f.read()


def _func_src(source, header):
    """切出一个函数/方法的源码（从 header 那行到**下一个同级或更外层定义**为止）。

    为什么要有它：K/I/J 那几组用 `"xxx" in src` 查整份文件，
    问的是"这个字符串在不在项目里"；但真正要锁的是**顺序**与**只出现在某个函数里**
    （例：`release(` 必须出现在采集线程的退出路径、却**不能**出现在 `stop()` 里）。
    拿整份文件查这类断言必然失真 —— 文件大了以后永远是 True，测试就变成了装饰。

    ⚠️ 分隔只找 `\\ndef` 是不够的：类里的方法之间全是缩进的 def，
       那样会把后面的方法一起切进来。所以按**缩进层级**判结尾。
    """
    lines = source.split("\n")
    start = next((n for n, ln in enumerate(lines) if header in ln), None)
    if start is None:
        raise AssertionError(f"源码里找不到「{header}」—— 重构后函数搬家/改名了，"
                             "先对着实现改这条锁，别删掉")
    ind = len(lines[start]) - len(lines[start].lstrip())
    body = [lines[start]]
    for ln in lines[start + 1:]:
        s = ln.strip()
        cur = (len(ln) - len(ln.lstrip())) if s else ind + 1
        if s and cur <= ind and s.startswith(("def ", "class ", "@")):
            break
        body.append(ln)
    return "\n".join(body)


CAM_SRC = _src(os.path.join("posture", "camera.py"))
DEPS_SRC = _src(os.path.join("posture", "deps.py"))
ENGINE_SRC = _src(os.path.join("posture", "engine.py"))
SETTINGS_SRC = _src(os.path.join("posture", "settings.py"))
COMPAT_SRC = _src(os.path.join("posture", "compat_matplotlib.py"))
# 界面（重构中：posture_app.py 已拆成 posture/ui/*）
UI_FILES = ["posture/ui/theme.py", "posture/ui/widgets.py", "posture/ui/review_panel.py",
            "posture/ui/parent_dialog.py", "posture/ui/main_window.py",
            "posture/ui/selftest.py", "posture/app.py"]
APP_SRC = "".join(_src(p) for p in UI_FILES if os.path.isfile(os.path.join(_HERE, p)))
# 兼容壳必须存在且真的转发（防止"壳被掏空、老入口静默罢工"）
SHIM_MON_SRC = _src("posture_monitor.py")
SHIM_ENG_SRC = _src("posture_engine.py")

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
    # ⚠️ 必须走 set_side_yaw()：SIDE_YAW_DEG 的真身在 posture.metrics（判定函数读它），
    #    直接写兼容壳的同名属性只会写进壳自己的字典 = **假开关**（改了但判定不生效）。
    pm.set_side_yaw(45.0)
    m45 = pm.compute_side_metrics(side(ear, sh), W16, H16)
    want45 = math.degrees(math.atan2(0.05 * W16 * math.sqrt(2), 0.30 * H16))
    close("B16 偏航 45° 下按 1/sin(yaw) 还原到矢状面 ≈ 22.74°",
          m45["neck_angle"], want45, 1e-6)
    case("B17 偏航 45° 测得角 > 正侧 90°（说明斜侧机会压低读数）",
         m45["neck_angle"] > m169["neck_angle"], True)
    case("B17b 读回的是 posture.metrics 的真值（写入口与读取口同一个地方）",
         pm.SIDE_YAW_DEG, 45.0)
    pm.set_side_yaw(90.0)

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
    src = ENGINE_SRC
    case("H1 引擎里已无「坐正->提示音」这个调用",
         "_speaker.chime(\"good\")" in src, False)
    case("H2 标定完成的提示音仍保留（那是系统事件，不是坐姿判定）",
         "_speaker.chime(\"done\")" in src, True)
    app_src = APP_SRC
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
    _src = CAM_SRC
    case("I2c open_camera 里 FOURCC 先于分辨率设置",
         _src.index("cap.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)") <
         _src.index("cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])"), True)
    case("I2d 构造期协商（带参数的 VideoCapture 构造函数）也在用 —— 省 4.3s/次开流",
         "cv2.VideoCapture(index, be, params)" in _src or
         "cv2.VideoCapture(i, be, params)" in _src, True)
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
    case("I15 界面的 _log() 接了落盘", "self.runlog.write(line)" in APP_SRC, True)


# ------------------------------------------------ J 组：SAC 拦截 matplotlib 的兜底
#
# 背景：本机 Windows 智能应用控制（SAC）拦掉了 matplotlib 的 _image.pyd，
#       mediapipe 的 vision 包因硬导入 matplotlib.pyplot 而整包失败，
#       表现为 `AttributeError: 'NoneType' object has no attribute 'PoseLandmarker'`。
#       修法是 compat_matplotlib.py 用桩顶掉 matplotlib（详见该模块文档）。
# 这组用例锁定：桩存在、幂等、不干扰真实 matplotlib、报错能带出真病因。

def test_sac_matplotlib_compat():
    print("\n[J] Windows SAC 拦截 matplotlib 的兜底")

    src = COMPAT_SRC
    case("J1 桩模块存在",
         os.path.isfile(os.path.join(_HERE, "posture", "compat_matplotlib.py")), True)
    case("J1b 根目录不再留重复副本（防两处漂移）",
         os.path.isfile(os.path.join(_HERE, "compat_matplotlib.py")), False)
    case("J2 桩提供 install()", "def install(" in src, True)
    case("J3 桩提供 stub_active()", "def stub_active(" in src, True)
    case("J4 桩顶掉的是 matplotlib.pyplot", 'sys.modules["matplotlib.pyplot"] = pyplot' in src, True)

    # 主模块必须真的调用它，而不是只把文件放那儿
    mon_src = DEPS_SRC
    case("J5 依赖入口调用 install()", "compat_matplotlib.install()" in mon_src, True)
    # 桩必须装在 mediapipe 之前（否则 mediapipe 已经导入失败了）
    case("J6 桩在 import mediapipe 之前装上",
         mon_src.index("compat_matplotlib.install()") < mon_src.index("import mediapipe as mp"), True)

    # 真正导入一遍，验证实际行为（不只做源码字符串匹配）
    here = _HERE
    sys.path.insert(0, here)
    try:
        from posture import compat_matplotlib as cm

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
    eng_src = ENGINE_SRC
    case("J13 引擎在 make_landmarker 前补了依赖守卫",
         "deps.need_mp()" in eng_src, True)
    case("J14 引擎的守卫在 make_landmarker 之前",
         eng_src.index("deps.need_mp()") < eng_src.index("pose.make_landmarker(model)"), True)


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

    src = CAM_SRC

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

    app = APP_SRC
    # ⚠️ 查的是"整个 UI 代码集合"：重构后界面拆成了 posture/ui/*，
    #    这几条锁的语义是**全局**的（整个界面里都不许出现旧写法），逐文件拼起来查才对。

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


# ------------------------------------------------- M. 摄像头子系统纯逻辑（重构后新增）
#
# 用户报的"加载很久"根因不在读帧慢，而在**协商方式**与**每次都重新探测能力**：
#   · `cap.open()` 之后再 `cap.set()` 谈格式+分辨率：本机实测 4.46s；
#     同样的参数交给**构造函数**一次谈完：0.17s —— 差 26 倍。
#   · 旧实现每次冷启动都要重跑一遍后端/档位探测（首次那几秒全花在这儿）。
# 这组只测**纯逻辑**：阶梯顺序、参数顺序、降档上报、本机档案读写、线程归属。
# 真机计时是 `--bench` 的事，不塞进离线自检（自检必须能在没摄像头的机器上跑）。

def test_camera_pure_logic():
    print("\n[M] 摄像头纯逻辑（档位阶梯 / 构造期协商 / 降档上报 / 本机档案 / 线程归属）")

    from posture import camera as cam
    import posture.engine as engine_mod
    src = CAM_SRC

    # ---- ① 档位阶梯：4K 优先，只许往下退
    lad = cam.tier_ladder((3840, 2160))
    case("M1 请求 4K 时第一档就是 4K（「4K 优先」的落点）", lad[0], (3840, 2160))
    case("M2 默认请求档就是 4K（CAM_TIERS[0]）", cam.CAM_TIERS[0], (3840, 2160))
    case("M3 阶梯高度严格递减", all(lad[i][1] > lad[i + 1][1]
                                   for i in range(len(lad) - 1)), True)
    case("M4 阶梯无重复（640x480 既在表里又被 append，只能出现一次）",
         len(set(lad)), len(lad))
    case("M5 兜底档 640x480 固定在最后", lad[-1], (640, 480))
    case("M6 阶梯里不存在高于请求档的档位（不许反过来偷偷升档）",
         [t for t in lad if t[1] > 2160], [])
    case("M7 请求 480p 时阶梯只有 480p（它下面没档了，也不该有）",
         cam.tier_ladder((640, 480)), [(640, 480)])
    odd = cam.tier_ladder((2592, 1944))
    case("M8 表外档位也从**请求档**起步（不是从表里挑一个）", odd[0], (2592, 1944))
    case("M8b 表外档位同样带兜底 480p", (640, 480) in odd, True)
    case("M8c 表外档位也不重复", len(set(odd)), len(odd))

    # ---- ② 构造期参数（"加载很久"的真正解药）
    if cam.cv2 is not None:
        p = cam._open_params((3840, 2160))
        case("M9 构造参数第一位是 FOURCC（格式必须先于分辨率）",
             p[0], cam.cv2.CAP_PROP_FOURCC)
        case("M9b 且值就是 MJPG", p[1], cam.MJPG_FOURCC)
        case("M10 宽度项先于高度项",
             p.index(cam.cv2.CAP_PROP_FRAME_WIDTH) < p.index(cam.cv2.CAP_PROP_FRAME_HEIGHT),
             True)
        case("M11 分辨率取自请求档且顺序为 W,H,W,H",
             p[2:6], [cam.cv2.CAP_PROP_FRAME_WIDTH, 3840,
                      cam.cv2.CAP_PROP_FRAME_HEIGHT, 2160])
        case("M12 顺手把缓冲压到 1 帧（否则重连后旧帧顶掉实时画面）",
             p[p.index(cam.cv2.CAP_PROP_BUFFERSIZE) + 1], 1)
    else:
        print("  SKIP  M9~M12（本机无 cv2）")
    case("M13 没给分辨率时**不**传构造参数（兜底轮用设备默认档）",
         cam._open_params(None), None)
    case("M13b 宽度为 0 同样不传", cam._open_params((0, 0)), None)

    s_sets = _func_src(src, "def _apply_sets(")
    case("M14 兜底 set 路径里 FOURCC 也先于宽度（顺序反了会静默失效）",
         s_sets.index("CAP_PROP_FOURCC") < s_sets.index("CAP_PROP_FRAME_WIDTH"), True)
    s_once = _func_src(src, "def open_once(")
    case("M15 open_once 首选构造期参数（0.17s vs 4.46s，实测 26 倍）",
         "cv2.VideoCapture(index, be, params)" in s_once, True)
    case("M16 只有后端不认构造参数时才退回 cap.set（MSMF 就是这种）",
         "except TypeError" in s_once and "_apply_sets(cap, res)" in s_once, True)

    probe = cam._probe_code("DSHOW", 0, 3840, 2160)
    case("M17 子进程探测也谈 MJPG（重构前漏了它，把能用的后端全误杀）",
         "cv2.CAP_PROP_FOURCC, mjpg" in probe, True)
    case("M18 探测脚本里格式也先于分辨率",
         probe.index("CAP_PROP_FOURCC") < probe.index("CAP_PROP_FRAME_WIDTH"), True)
    case("M19 探测走构造期参数（与真实开流同一套协商方式，结论才可比）",
         "cv2.VideoCapture(0, cv2.CAP_DSHOW, [" in probe, True)
    case("M20 后端不认构造参数时探测自己降级重试",
         "except TypeError" in probe, True)
    case("M21 探测按**像素内容**判可用（只看 read 的 ok 会被全零帧骗）",
         "best > 8" in probe, True)

    # ---- ③ 降档不许静默：OpenResult 是界面唯一的依据
    r_ok = cam.OpenResult(cap=None, camera=0, backend="DSHOW", width=3840, height=2160,
                          fourcc="MJPG", elapsed=1.2, attempts=[{"backend": "DSHOW"}],
                          degraded=False, requested=(3840, 2160), warnings=[])
    case("M22 未降档时 summary 不含「已降级」", "已降级" in r_ok.summary(), False)
    case("M22b MJPG 谈到了 -> mjpg=True", r_ok.mjpg, True)
    r_deg = cam.OpenResult(cap=None, camera=0, backend="DSHOW", width=1920, height=1080,
                           fourcc="YUY2", elapsed=3.4,
                           attempts=[{"backend": "DSHOW"}, {"backend": "MSMF"}],
                           degraded=True, requested=(3840, 2160),
                           warnings=["只拿到 1920x1080，低于请求的 3840x2160。"])
    case("M23 降档必须写进 summary", "已降级" in r_deg.summary(), True)
    case("M23b 并且带上**请求档**（家长要能看出差了多少）",
         "3840x2160" in r_deg.summary(), True)
    case("M23c 尝试次数也在（排查时决定要不要换后端）", "尝试=2 次" in r_deg.summary(), True)
    case("M24 没协商到 MJPG 时 mjpg=False（界面要据此报警）", r_deg.mjpg, False)
    wl = ["a"]
    r_w = cam.OpenResult(cap=None, camera=0, backend="DSHOW", width=640, height=480,
                         fourcc="MJPG", elapsed=0.5, attempts=[], degraded=True,
                         requested=(3840, 2160), warnings=wl)
    wl.append("b")
    case("M25 warnings 存的是拷贝（外部改列表不会篡改已上报的警告）",
         len(r_w.warnings), 1)
    case("M25b YUY2/MJPG1 之外的写法也算没谈到 MJPG",
         [cam._mjpg_ok(x) for x in ("YUY2", "MJP1", None, "mjpg")],
         [False, True, False, True])

    # ---- ④ 报错分两段：短文案上界面，清单进详情
    e1 = cam.CameraError("摄像头打不开", "排查顺序：\n  1) 是否被别的程序占用")
    case("M26 CameraError 带短文案 + 详情两段",
         (e1.short, "占用" in e1.detail), ("摄像头打不开", True))
    case("M27 str() 同时含两段（CLI 直接 print 也不丢信息）",
         all(x in str(e1) for x in ("摄像头打不开", "占用")), True)
    case("M27b 没有详情时只返回短文案", str(cam.CameraError("只有一句")), "只有一句")
    case("M28 排查清单覆盖占用/隐私/USB/档位/后端五个方向",
         [k in cam.CHECKLIST for k in ("占用", "隐私", "USB", "档位", "后端")],
         [True] * 5)

    # ---- ⑤ 本机档案（camera_profile.json）：记住上次成功，但只做"先试哪个"
    tmpd = tempfile.mkdtemp(prefix="posture_profile_")
    try:
        pth = os.path.join(tmpd, "camera_profile.json")
        pr = cam.CameraProfile(path=pth)
        case("M29 新档案默认带 version/last_good/probes/tiers 四个字段",
             sorted(pr.d), ["last_good", "probes", "tiers", "version"])
        pr.set_last_good("DSHOW", 0, 3840, 2160, "MJPG", 1.23)
        again = cam.CameraProfile.load(path=pth)
        case("M30 写进磁盘后重新读回仍命中",
             again.matches_last_good("DSHOW", 0, 3840, 2160), True)
        case("M31 换了档位就不再命中（换机器/改分辨率不会被旧档案带偏）",
             again.matches_last_good("DSHOW", 0, 1920, 1080), False)
        case("M31b 换了后端也不再命中",
             again.matches_last_good("MSMF", 0, 3840, 2160), False)
        case("M31c 实测开流耗时进了档案（--bench 才能对比新旧实现）",
             again.last_good["opened_sec"], 1.23)
        again.set_probe("DSHOW|0|3840x2160", True)
        again.set_tier("DSHOW", 0, 3840, 2160, True, actual=[3840, 2160], note="MJPG 可靠")
        r3 = cam.CameraProfile.load(path=pth)
        case("M32 探测结果落盘后能读回（第二次启动不再付子进程代价）",
             r3.probe("DSHOW|0|3840x2160"), True)
        case("M32b 没探测过的组合返回 None 而不是 False（'不知道'≠'不行'）",
             r3.probe("MSMF|0|3840x2160"), None)
        case("M32c 档位实测结论也能读回", r3.tier("DSHOW", 0, 3840, 2160)["usable"], True)
        case("M33 save 走原子替换（不留 .tmp，断电不会剩半个 JSON）",
             os.path.exists(pth + ".tmp"), False)
        case("M33b 落盘的是合法 UTF-8 JSON",
             json.load(open(pth, encoding="utf-8"))["version"], cam.CameraProfile.VERSION)
        pr.forget()
        f = cam.CameraProfile.load(path=pth)
        case("M34 forget() 一次清空三类记忆（家长点「重新探测摄像头」）",
             (f.last_good, f.probe("DSHOW|0|3840x2160"), f.tier("DSHOW", 0, 3840, 2160)),
             (None, None, None))
        with open(pth, "w", encoding="utf-8") as fh:
            fh.write("{ 这不是合法 JSON")
        bad = cam.CameraProfile.load(path=pth)
        case("M35 档案坏了就当没有（绝不能让程序起不来）", bad.last_good, None)
        bad.set_probe("x", True)
        case("M35b 坏了的档案还能重写回（下一次就是合法 JSON）",
             json.load(open(pth, encoding="utf-8"))["probes"]["x"], True)
        case("M36 档案只写在本机路径下（临时目录里没有第二份）",
             sorted(os.listdir(tmpd)), ["camera_profile.json"])
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    case("M37 档案默认路径是本机的 camera_profile.json",
         os.path.basename(cam.paths.CAMERA_PROFILE_PATH), "camera_profile.json")
    gi = os.path.join("..", "..", ".gitignore")
    git = _src(gi) if os.path.isfile(os.path.join(_HERE, gi)) else ""
    case("M38 本机档案/基线/配置/审核图都被 git 忽略（孩子照片只留本机）",
         [k in git for k in ("camera_profile.json", "baseline.json",
                             "config.json", "review/")], [True] * 4)

    # ---- ⑨ 采集线程归属（DSHOW=COM：创建与读取必须同线程，跨线程创建实测 0/5 全废）
    s_run = _func_src(src, "def _run(self):")
    case("M40 开流发生在采集线程内（句柄绑创建线程）", "open_camera_result(" in s_run, True)
    case("M41 稳态读取用**不带超时**的 read_frame（每帧都起线程没意义）",
         "read_frame(self._cap)" in s_run and "read_frame_timed(" not in s_run, True)
    case("M42 掉线靠心跳判（read 返回 False 不等于掉线）", "STALL_SEC" in s_run, True)
    case("M42b 读到新帧会把状态从 reconnecting 拉回 ready",
         '_set(_phase="ready"' in s_run, True)
    case("M43 线程退出时由**它自己** release（finally 里）",
         "finally:" in s_run and "_release()" in s_run, True)
    s_stop = _func_src(src, "def stop(self):")
    case("M44 stop() 只发停止请求，绝不跨线程 release 句柄", "release(" not in s_stop, True)
    case("M44b stop() 会等采集线程（它可能正阻塞在 read 上）", "join(" in s_stop, True)
    s_latest = _func_src(src, "def latest(self):")
    case("M45 latest() 不消费帧槽（判定每拍都要拿最新帧）",
         "self._seen" not in s_latest, True)
    s_poll = _func_src(src, "def poll(self):")
    case("M45b 只有 poll() 做「只看新帧」（界面省一次图像转换）",
         "self._seen = seq" in s_poll, True)
    s_reopen = _func_src(src, "def _reopen(self):")
    case("M46 重开在**同一线程**里做（恢复出来的句柄才合法）",
         "open_camera_result(" in s_reopen, True)
    case("M47 重开失败按 REOPEN_DELAY_SEC 退避（设备被抢时不疯狂重试）",
         "REOPEN_DELAY_SEC" in s_reopen, True)
    case("M48 重连成功累加次数（引擎据此才发现档位变了、要重校验基线）",
         "_reopen_count += 1" in s_reopen, True)
    case("M49 重连后把 warnings 一起换掉（旧降档警告不能常驻）",
         "self._warns = list(res.warnings)" in s_reopen, True)

    # ---- ⑩ 预算常量本身（这几个数字就是"加载很久"的上限）
    case("M50 开流总预算存在且落在 5~20s（可以慢，但不许无限）",
         5.0 <= cam.OPEN_BUDGET_SEC <= 20.0, True)
    case("M51 子进程探测超时 < 总预算（一个后端不能吃光预算）",
         cam.OPEN_PROBE_TIMEOUT < cam.OPEN_BUDGET_SEC, True)
    case("M52 掉线阈值 3~15s（4K 帧间隔 ~0.07s，设太小会误判掉线）",
         3.0 <= cam.STALL_SEC <= 15.0, True)
    case("M53 重连退避 < 掉线阈值（判掉线后别干等）",
         cam.REOPEN_DELAY_SEC < cam.STALL_SEC, True)
    case("M54 同档重开次数 >= 3（本机 1080p 实测成功率只有 5/8）",
         cam.MAX_OPEN_ATTEMPTS >= 3, True)

    # ---- ⑪ 预算是**准入闸**，不是"事后截断"
    #   `deadline` 本身只在"每次读帧之前"起作用，管不住一轮尝试的**起点**：
    #   真机验收里出现过「预算 15s、22.35s 才报错」——起点的剩余预算已经只够
    #   开一次流，却还去探 MSMF + 等 6 张全零帧。所以剩余不足就别开始。
    s_open = _func_src(src, "def open_camera_result(")
    case("M55 存在准入闸常量（1.5~4s：太小根本开不完一次，太大白丢预算）",
         1.5 <= cam.MIN_ATTEMPT_SEC <= 4.0, True)
    case("M55b 准入闸必须小于总预算（否则预算形同没有）",
         cam.MIN_ATTEMPT_SEC < cam.OPEN_BUDGET_SEC, True)
    case("M56 准入闸判**两处**：外层 attempt 循环 + 内层每个候选组合",
         s_open.count("remain < MIN_ATTEMPT_SEC"), 2)
    case("M57 一档试完也用准入闸收尾（不然阶梯会一路往下退到 480p）",
         "if deadline - time.perf_counter() < MIN_ATTEMPT_SEC:" in s_open, True)
    case("M58 子进程探测超时被剩余预算夹住（一个探测不能吃光预算）",
         "timeout=min(OPEN_PROBE_TIMEOUT, max(1.0, remain))" in s_open, True)
    case("M59 单次开流的读帧超时同样夹剩余预算，且 deadline 一路传进 open_once",
         "timeout=min(2.5, max(0.5, remain))" in s_open
         and "deadline=deadline" in s_open, True)

    # ---- ⑫ 兼容壳：老入口必须**真的**还能按名字取到这些常量
    #   壳里只有 import，所以查壳的源码字符串只防"名字被删"；真正防的是
    #   "壳自己抄了一份常量" —— 那样改了包内数值，壳还留着旧数，属于静默罢工。
    #   用 **is** 判同一对象才作数。
    case("M60 壳 posture_monitor 源码仍转发采集预算常量",
         [k in SHIM_MON_SRC for k in ("OPEN_BUDGET_SEC", "MIN_ATTEMPT_SEC",
                                      "MAX_OPEN_ATTEMPTS", "STALL_SEC")], [True] * 4)
    case("M60b 壳取到的是包内**同一个对象**（不是抄的一份，改了数值不会两头不一致）",
         [getattr(pm, k) is getattr(cam, k) for k in ("OPEN_BUDGET_SEC",
                                                      "MIN_ATTEMPT_SEC", "STALL_SEC")],
         [True] * 3)
    case("M61 壳 posture_engine 仍转发引擎与周边（老脚本 pe.PostureEngine / pe.RunLog 还在）",
         [getattr(pe_mod, k, None) is not None for k in
          ("PostureEngine", "RunLog", "ReviewStore", "Speaker", "apply_config")],
         [True] * 5)
    case("M61b 引擎壳转发的也是包内同一个对象",
         pe_mod.PostureEngine is engine_mod.PostureEngine, True)

    # ---- ⑬ 开流进度必须能递到界面
    #   界面上那句「已经等了 8 秒，正在试 MSMF ／ 3840x2160 第 2 轮」不 decoration：
    #   同样是等，能不能看出"卡在探测/卡在坏开流/卡在等有效帧"，决定了家长会不会
    #   以为程序死了（这正是重构前"加载很久"最难排查的一半）。
    case("M62 open_camera_result 支持 on_progress，并在每个候选组合上调用",
         "on_progress=None" in s_open and "progress(name, i, res, attempt)" in s_open, True)
    s_pr = s_open.index("progress(name, i, res, attempt)")
    s_pb = s_open.index("backend_viable_subprocess(")
    s_oc = s_open.index("open_once(i, name, res")
    case("M63 报名在子进程探测**之前**（探测动辄几秒，界面得先知道在试谁）",
         s_pr < s_pb < s_oc, True)
    s_run2 = _func_src(src, "def _run(self):")
    s_reopen2 = _func_src(src, "def _reopen(self):")
    case("M64 采集线程把 on_progress 接到 _note_trying",
         "on_progress=self._note_trying" in s_run2, True)
    case("M64b 重连走**同一条**进度路径（掉线时界面同样有话可说）",
         "on_progress=self._note_trying" in s_reopen2, True)
    case("M65 成功/失败都要把 trying 清空（否则状态条永远挂着「正在试」）",
         "_trying=None" in s_run2 and "self._trying = None" in s_reopen2, True)
    s_status = _func_src(src, "def status(self):")
    case("M66 status() 把 trying 交出去（引擎只能从这个口子拿）",
         '"trying"' in s_status, True)
    case("M67 引擎的 _cam_view 转发 trying（漏一个键，界面就只能干等）",
         '"trying"' in ENGINE_SRC, True)

    # ---- ⑭ fourcc 回读：宁可少信息，不许吐乱码
    s_fc = _func_src(src, "def _fourcc_str(")
    case("M68 fourcc 只接受可打印 ASCII（真机出现过 =}ë6ä 这种内部数值）",
         "0x20 <= ord(c) < 0x7F" in s_fc, True)
    case("M68b 认不出来时给「未知 + 原始值」，父级仍能据此排查",
         "未知(" in s_fc, True)
    case("M68c 设备把 fourcc 报成 0 时不显示成 8 个零",
         "if v else" in s_fc, True)


# ------------------------------------------------- N. 引擎 × 设备状态机（假采集线程）
#
# 这组不开任何设备：把 `CameraStream`、`pose.detect`、`pose.make_landmarker`、
# `find_model`、依赖守卫、基线读写全部换成假的，用真 `PostureEngine` 跑一遍状态机。
# 锁的是重构新增的那几条契约：
#   ① 开流期间快照里就有 cam（界面才画得出"正在试 DSHOW / 3840x2160"）；
#   ② 请求 4K 只拿到 1080p -> degraded + cam_warnings 非空（**不许静默降档**）；
#   ③ 基线按**实际**分辨率校验：4K 标的基线在 1080p 下作废重标（防「低头」漏报）；
#   ④ 中途断线 -> 统计与判定**都暂停**，且不许显示成"看不到你啦"；
#   ⑤ 重连换档 -> 自动重标；同档重连 -> 不许白白重标。
# ⚠️ 它必须在最后跑：引擎会 `settings.apply_config(cfg)` 改判定用的模块全局。

class _FakeLM:
    def close(self):
        pass


class _FakeStream:
    """假采集线程：与 `camera.CameraStream` 同形，只按脚本切 phase / 帧槽。"""

    def __init__(self, res, open_delay=0.2):
        self._res = res
        self._delay = open_delay
        self.budget = 15.0
        self.phase = "opening"
        self.reconnects = 0
        self.stop_called = False
        self._seq = 0
        self._frame = None
        self._short = None
        self._detail = None
        # 引擎**只准**调用它一次。多出来的每一次都意味着"程序自己在背后重试"，
        # 而我们的口径是：启动失败要把原因摊开、由人点「重试」（见 README/CHANGELOG）。
        self.start_count = 0
        self._t = time.perf_counter()

    def start(self):
        self.start_count += 1
        self._t = time.perf_counter()

    def stop(self):
        self.stop_called = True

    def status(self):
        if self.phase == "opening" and time.perf_counter() - self._t > self._delay:
            self.phase = "ready"
        return {"phase": self.phase, "backend": self._res.backend,
                "camera": self._res.camera, "width": self._res.width,
                "height": self._res.height, "degraded": self._res.degraded,
                "reconnects": self.reconnects, "short": self._short,
                "detail": self._detail,
                "stale_sec": 0.1, "warnings": list(self._res.warnings),
                "result": self._res.summary(), "seq": self._seq,
                # 开流期间真采集线程会给"正在试哪个组合"（见 open_camera_result
                # 的 on_progress）。假线程也要给同样的形状，界面那句进度才不是空话。
                "trying": ({"backend": self._res.backend, "camera": self._res.camera,
                            "width": self._res.requested[0],
                            "height": self._res.requested[1], "attempt": 2}
                           if self.phase in ("opening", "reconnecting") else None)}

    @property
    def result(self):
        return self._res if self.phase in ("ready", "reconnecting") else None

    def latest(self):
        return self._frame, self._seq

    # ---- 测试脚本动作
    def feed(self, fr):
        self._frame = fr
        self._seq += 1

    def drop(self):
        self.phase = "reconnecting"
        self._short = "已经 6 秒没有新画面"

    def fail(self, short, detail):
        """开流彻底失败（真线程里是 phase="failed" + 排查清单）。

        必须在 `with _EngineRun(...)` **之前**调用：phase 一旦不是 "opening"，
        `status()` 里那条"到点就转 ready"的脚本就不会触发，引擎必然走进错误分支。
        """
        self.phase = "failed"
        self._short = short
        self._detail = detail

    def restore(self, res=None):
        if res is not None:
            self._res = res
        self.phase = "ready"
        self.reconnects += 1
        self._short = None


def _res(req_w, req_h, got_w=None, got_h=None, warns=(), fourcc="MJPG"):
    """一份"请求 X、实际拿到 Y"的开流结果（不碰任何设备）。"""
    from posture import camera
    gw, gh = (got_w or req_w), (got_h or req_h)
    return camera.OpenResult(cap=None, camera=0, backend="DSHOW",
                           width=gw, height=gh, fourcc=fourcc, elapsed=0.9,
                           attempts=[{"backend": "DSHOW", "camera": 0, "width": req_w,
                                      "height": req_h, "result": "ok", "sec": 0.9}],
                           degraded=(gw, gh) != (req_w, req_h),
                           requested=(req_w, req_h), warnings=list(warns))


class _EngineRun:
    """在补丁窗口内起一次真引擎；退出时保证补丁、模块全局、临时文件全部还原。"""

    def __init__(self, res, stream=None, seed=(3840, 2160), calib_sec=0.3, fps=40):
        self.res, self.stream_pre = res, stream
        self.seed, self.calib_sec, self.fps = seed, calib_sec, fps
        self.saved, self.made, self.snaps = [], {}, []
        self.seq = [{"neck_angle": 8.0, "trunk_angle": 6.0, "sh_ear_len": 210.0}
                    for _ in range(4000)]
        self.tmpd = tempfile.mkdtemp(prefix="posture_engine_")
        self.base_path = os.path.join(self.tmpd, "baseline.json")
        self.eng = self.stream = None
        self.t0 = None

    def __enter__(self):
        import posture.camera as C
        import posture.engine as E
        import posture.pose as P
        import posture.settings as S
        self.C, self.E, self.P, self.S = C, E, P, S
        self.base = dict(S.DEFAULT_CONFIG)
        self.real = (C.CameraStream, P.detect, P.make_landmarker,
                     E.find_model, E.deps.need_mp,
                     S.load_baseline, S.save_baseline, S.baseline_reason)
        # metrics 的模块全局会被 apply_config 改写 —— 先原样存一份
        import posture.metrics as MX
        self.MX = MX
        self.globals = (dict(MX.THRESHOLDS), dict(MX.SIDE_THRESHOLDS),
                        MX.SIDE_YAW_DEG, S.FPS_SAMPLE)

        # 预置一份"上次在 self.seed 档标好的基线"，用来验基线与实际分辨率绑定
        self.real[6]({"camera": 0, "side_yaw": 90.0, "width": self.seed[0],
                      "height": self.seed[1], "model": "lite"},
                     dict(self.seq[0]), self.base_path)

        def ctor(idx=None, backend=None, width=None, height=None,
                 budget=None, on_event=None, name="camera"):
            self.made["kw"] = {"idx": idx, "width": width, "height": height}
            # 数一下引擎到底 new 了几个采集线程对象 —— 这是"有没有偷偷自动重试"的铁证
            self.made["ctor_calls"] = self.made.get("ctor_calls", 0) + 1
            self.made["stream"] = self.stream_pre or _FakeStream(self.res)
            return self.made["stream"]

        def mk(path):
            """假的"建模型"：记下它发生的时刻与**当时流还在什么阶段**。

            N7 要证的就是"模型构造发生在开流期间"（重构前是串行的，白等 0.3~1s）。
            """
            self.made["lm_at"] = time.perf_counter() - self.t0
            st = self.made.get("stream")
            self.made["lm_phase"] = st.phase if st else "(还没建流)"
            return _FakeLM()

        def detect(lm, fr, t_sec, fn):
            return (self.seq.pop(0) if self.seq else None), None

        C.CameraStream = ctor
        P.detect = detect
        P.make_landmarker = mk
        E.find_model = lambda p: "fake-task-model"
        E.deps.need_mp = lambda: None
        S.load_baseline = lambda cfg, path=None: self.real[5](cfg, self.base_path)
        S.baseline_reason = lambda cfg, path=None: self.real[7](cfg, self.base_path)

        def save(cfg, base, path=None):
            self.saved.append({"camera": cfg.get("camera"), "side_yaw": cfg.get("side_yaw"),
                               "width": cfg.get("width"), "height": cfg.get("height")})
            return self.real[6](cfg, base, self.base_path)

        S.save_baseline = save

        cfg = dict(self.base)
        cfg.update({"camera": 0, "side_yaw": 90.0,
                    "width": self.res.requested[0], "height": self.res.requested[1],
                    "calib_sec": self.calib_sec, "fps_sample": self.fps,
                    "review_enabled": False, "review_sample_sec": 0, "mute": True})
        self.t0 = time.perf_counter()
        self.eng = E.PostureEngine(cfg)
        self.eng.start()
        for _ in range(400):                       # CameraStream 在引擎线程里才 new
            if "stream" in self.made:
                break
            time.sleep(0.005)
        self.stream = self.made.get("stream")
        if self.stream is None:
            return self          # 引擎连采集线程都没起来（后面用例会红得明明白白）
        # 等进监测（或出错）
        dl = time.perf_counter() + 8.0
        while time.perf_counter() < dl:
            self.stream.feed(_FRAME)
            s = self.eng.snapshot()
            last = self.snaps[-1] if self.snaps else None
            if last is None or (s["phase"], s["level"], s["text"]) != \
                    (last["phase"], last["level"], last["text"]):
                self.snaps.append(s)
            if s["phase"] in ("monitoring", "error", "stopped"):
                break
            time.sleep(0.02)
        return self

    def __exit__(self, *exc):
        try:
            if self.eng:
                self.eng.stop()
                self.eng.join(timeout=3)
            if self.stream is not None:
                self.stream_stop_called = self.stream.stop_called
        finally:
            (self.C.CameraStream, self.P.detect, self.P.make_landmarker,
             self.E.find_model, self.E.deps.need_mp,
             self.S.load_baseline, self.S.save_baseline,
             self.S.baseline_reason) = self.real
            (self.MX.THRESHOLDS, self.MX.SIDE_THRESHOLDS,
             self.MX.SIDE_YAW_DEG, self.S.FPS_SAMPLE) = self.globals
            shutil.rmtree(self.tmpd, ignore_errors=True)
        return False

    def feed_until(self, pred, timeout=4.0):
        """持续喂帧直到快照满足 pred；返回最后一份快照。"""
        dl = time.perf_counter() + timeout
        s = self.eng.snapshot()
        while time.perf_counter() < dl:
            self.stream.feed(_FRAME)
            s = self.eng.snapshot()
            if pred(s):
                break
            time.sleep(0.012)
        return s


_FRAME = _FakeFrame(200)      # 1280 宽 = PREVIEW_MAX_W，pose.shrink 原样返回，不需要 numpy
_WARN_A = "只拿到 1920x1080，低于请求的 3840x2160。"


def test_engine_device_state():
    print("\n[N] 引擎 × 设备状态机（假采集线程，验证降档上报/基线绑定/断线暂停）")

    import posture.engine as E
    import posture.texts as T
    from posture import camera

    # ---------- 场景 A：请求 4K 却只拿到 1080p
    with _EngineRun(_res(3840, 2160, 1920, 1080, warns=[_WARN_A])) as R:
        ph = [s["phase"] for s in R.snaps]
        case("N1 先进 opening、最后进 monitoring",
             ("opening" in ph, ph[-1]), (True, "monitoring"))
        case("N2 开流期间快照里就有 cam（界面才画得出进度）",
             any((s.get("cam") or {}).get("phase") == "opening" for s in R.snaps), True)
        case("N3 降档必须上报 degraded（不许静默换成低档）",
             any((s.get("cam") or {}).get("degraded") for s in R.snaps), True)
        case("N4 降档警告原文进了 cam_warnings",
             any(_WARN_A in (s["cam_warnings"] or []) for s in R.snaps), True)
        case("N5 交给采集线程的是**请求档** 3840x2160",
             (R.made["kw"]["width"], R.made["kw"]["height"]), (3840, 2160))
        case("N6 实际档位回填进 _eff（基线按它校验）",
             (R.eng._eff.get("width"), R.eng._eff.get("height")), (1920, 1080))
        case("N7 模型构造与开流并行（建模型时流还没开好）",
             (R.made.get("lm_phase"), R.made.get("lm_at", 9) < 0.2), ("opening", True))
        case("N8 4K 标的旧基线在 1080p 下作废、自动重标",
             [x["width"] for x in R.saved], [1920])
        case("N8b 新基线按**实际**分辨率落盘（不是请求值）",
             all(x["width"] == 1920 and x["height"] == 1080 for x in R.saved), True)
        st = R.feed_until(lambda s: s["level"] == "good")
        case("N9 重标后回到正常监测并判为良好", (st["phase"], st["level"]),
             ("monitoring", "good"))
        case("N10 快照里的 cam 是实际档位",
             (st["cam"]["width"], st["cam"]["height"]), (1920, 1080))
        # 界面上那句「已经等了 N 秒，正在试 DSHOW ／ 3840x2160 第 2 轮」的数据源
        case("N10b 开流期间快照给出「正在试哪个组合」（trying）",
             any((s.get("cam") or {}).get("trying") for s in R.snaps), True)
        case("N10c 拿到画面后 trying 必须清空（不然界面永远挂着「正在试…」）",
             all(s["cam"].get("trying") is None for s in R.snaps
                 if s["phase"] == "monitoring" and s.get("cam")), True)

    # ---------- 场景 B：断线 -> 自动重连 -> 恢复（档位没变）
    streamB = _FakeStream(_res(3840, 2160), open_delay=0.05)
    with _EngineRun(_res(3840, 2160), stream=streamB, seed=(3840, 2160)) as R:
        R.feed_until(lambda s: s["level"] == "good")
        s1 = R.eng.snapshot()
        case("N11 采集条件与基线一致时**不**重标（不白付 10 秒）", R.saved, [])
        case("N11b 基线读回进了快照", bool(s1["baseline"]), True)
        case("N12 正常监测时统计在增长", s1["stats"]["total_sec"] > 0, True)
        t_before = s1["stats"]["total_sec"]
        streamB.drop()
        s2 = R.feed_until(lambda s: (s["cam"] or {}).get("phase") == "reconnecting")
        case("N13 断线期间统计**不再**增长（漏报误报都源自这里）",
             abs(s2["stats"]["total_sec"] - t_before) < 0.06, True)
        case("N14 断线时明确告知正在自动重连", s2["text"], T.TEXT_RECONNECTING)
        case("N15 断线时 phase 仍是 monitoring（不退化成错误态）",
             s2["phase"], "monitoring")
        case("N16 断线**不**判成「看不到你啦」（那是设备问题，不是孩子的问题）",
             s2["level"] != "nodetect", True)
        case("N16b 断线时快照给出重连状态与中断时长（界面才有话可说）",
             (s2["cam"]["phase"], s2["cam"]["stale_sec"] is not None), ("reconnecting", True))
        case("N16c 重连期间也要给出「正在试哪个组合」（和首次开流同一条进度路径）",
             bool(s2["cam"]["trying"]), True)
        streamB.restore()
        s3 = R.feed_until(lambda s: s["level"] == "good")
        case("N17 重连后恢复判定与统计", s3["stats"]["total_sec"] > t_before, True)
        case("N18 重连后文案回到正常状态",
             s3["text"] in (T.TEXT_GOOD, T.TEXT_NODETECT), True)
        case("N19 同档重连不算「采集条件变了」（不该白白重标）", R.saved, [])

    # ---------- 场景 C：重连后档位变了（4K -> 1080p）
    streamC = _FakeStream(_res(3840, 2160), open_delay=0.05)
    with _EngineRun(_res(3840, 2160), stream=streamC, seed=(3840, 2160)) as R:
        R.feed_until(lambda s: s["level"] == "good")
        case("N20 起始 4K + 4K 基线：直接复用", R.saved, [])
        streamC.drop()
        streamC.restore(_res(3840, 2160, 1920, 1080, warns=[_WARN_A]))
        s = R.feed_until(lambda x: (x["cam"] or {}).get("width") == 1920)
        case("N21 引擎发现重连后档位变了并回填快照",
             (R.eng._eff.get("width"), s["cam"]["width"]), (1920, 1920))
        # ⚠️ 这里不能用 feed_until(level==good)：重标要 0.4s，等结果会把
        #    "曾经回到标定中"这一瞬冲掉 —— 而"作废重标到底有没有发生"正是要验的东西。
        saw_calib = False
        dl = time.perf_counter() + 8.0
        while time.perf_counter() < dl:
            R.stream.feed(_FRAME)
            s = R.eng.snapshot()
            if s["phase"] == "calibrating":
                saw_calib = True
            if saw_calib and R.saved:
                break
            time.sleep(0.012)
        case("N22 档位一变立刻作废基线、界面回到「标定中」", saw_calib, True)
        case("N23 重标的新基线按**实际**档位落盘（不是请求的 4K）",
             [(x["width"], x["height"]) for x in R.saved], [(1920, 1080)])
        s = R.feed_until(lambda x: x["level"] == "good")
        case("N24 降档警告同步给了界面（重连后也要说）",
             any(_WARN_A in w for w in (s["cam_warnings"] or [])), True)
        case("N25 重连后正常监测（不需要重启程序）", s["phase"], "monitoring")
    # `__exit__` 之后才看得到"引擎有没有把停止请求转给采集线程"
    case("N26 引擎 stop() 会把停止请求转给采集线程（不自己关设备）",
         streamC.stop_called, True)
    case("N26b 各场景结束时采集线程都收到了停止请求", streamB.stop_called, True)

    # ---------- 源码级：结构契约（这几条一破，界面就又回到"干等/静默"）
    es = ENGINE_SRC
    s_adopt = _func_src(es, "def _adopt_camera(")
    case("N27 降档结果**不**写回 config.json（下次仍然先试 4K）",
         "save_config" in s_adopt, False)
    s_run = _func_src(es, "def run(self):")
    case("N28 采集线程先起、模型后建（两条耗时路径并行）",
         s_run.index("camera.CameraStream(") < s_run.index("pose.make_landmarker(model)"),
         True)
    case("N29 开流期间持续把 cam 写进快照（界面进度条的数据源）",
         "self._set(cam=self._cam_view(st))" in _func_src(es, "def _await_camera("), True)
    case("N30 等摄像头有外部上限（开流预算 + 余量，不许无限等）",
         getattr(E, "CAM_WAIT_EXTRA_SEC", 0) > 0, True)
    s_stop = _func_src(es, "def stop(self):")
    case("N31 引擎 stop() 不自己 release 设备句柄（COM 归采集线程）",
         "release(" not in s_stop, True)
    case("N32 断线分支排在统计累加之前（没画面时一拍都不许算进统计）",
         es.index('if cam_phase != "ready"') < es.index('stats["total_sec"] += dt'), True)
    case("N33 监测里以**实际**条件校验基线（_camera_moved -> load_baseline）",
         "_camera_moved()" in _func_src(es, "def _monitor("), True)
    case("N34 采集条件变化时重新校验基线（不是继续用旧的）",
         "settings.load_baseline(self._eff)" in es, True)

    # ---------- 场景 D：启动就开不出画面 —— 锁的是"**不**自动重试"
    # ⚠️ 为什么要把"没有重试"写成用例：这是 2026-10-18 定下的一条**设计决定**
    #    （README「摄像头子系统」/ CHANGELOG §F）。实测里最恶劣的一类失败是
    #    "设备刚被另一个进程放开着、驱动还在释放"，此时 UVC 会回一堆全 0 的占位帧。
    #    程序自己在背后连开三次的结果是：家长看到三个不同的错误、孩子看到黑屏，
    #    而真正管用的是"把原因摊开 + 过几秒点一下重试"。
    #    注意分工：**开成功之后**的掉线才归自动重连管（场景 B 锁的就是那条路径）；
    #    开流本身失败归人管。两边都锁住，将来谁改动这条分工都会红。
    streamD = _FakeStream(_res(3840, 2160), open_delay=0.05)
    why_short = "摄像头打不开：DSHOW / MSMF 全部档位都没有有效画面"
    streamD.fail(why_short, camera.CHECKLIST)
    with _EngineRun(_res(3840, 2160), stream=streamD) as R:
        s = R.eng.snapshot()
        case("N35 开流失败要落到 error（不许卡在 opening 干等）", s["phase"], "error")
        case("N36 采集线程的短文案原样进 error（界面第一行就显示它）",
             s["error"], why_short)
        case("N37 排查清单进 error_detail（收进「详情」，不糊在孩子界面上）",
             "排查顺序" in (s["error_detail"] or ""), True)
        case("N38 失败后引擎**不**重建采集线程（CameraStream 只被 new 一次）",
             R.made.get("ctor_calls"), 1)
        case("N38b 失败后引擎**不**重启采集线程（start 也只被调用一次）",
             streamD.start_count, 1)
        # 错误分支自己不该藏着 .start()：重试的唯一入口是界面上那个按钮
        # （main_window._on_toggle → _restart，整条路径重走一遍，配置还是 4K 优先）。
        s_err = es.split("except camera.CameraError as e:")[1].split("except SystemExit")[0]
        case("N39 错误分支内不重启采集（重试归人，见 README）", ".start(" in s_err, False)
    case("N40 失败退出时把设备让出来（stop 采集线程，重试才开得动）",
         streamD.stop_called, True)


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
    test_camera_pure_logic()
    test_engine_device_state()      # ★ 必须最后跑：引擎会改判定用的模块全局

    print("\n" + "=" * 62)
    if FAILS:
        print(f"结果：{COUNT - len(FAILS)}/{COUNT} 通过，{len(FAILS)} 个失败")
        for f in FAILS:
            print(f"  ✗ {f}")
        sys.exit(1)
    print(f"结果：{COUNT}/{COUNT} 全部通过 ✅")


if __name__ == "__main__":
    main()

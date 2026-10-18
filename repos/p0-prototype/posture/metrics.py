# -*- coding: utf-8 -*-
"""坐姿指标与判定：正面 / 侧面 / 融合 / 合理性筛除 / 阈值建议 / 标定自检。

纯计算模块 —— 不碰摄像头、不碰界面、不 import 任何第三方库。
唯一的外部依赖是 `posture.texts` 里的问题名（避免两处各写一遍中文）。

为什么这些数字住在这里（而不是 settings）：
    THRESHOLDS / SIDE_THRESHOLDS / SANE / SIDE_SANE / SIDE_YAW_DEG 会被
    `settings.apply_config()` 在运行时**原地改写**（判定函数读的是模块全局，
    这是既有设计；改签名会牵动 156 条离线回归用例）。
    "会被改写的判定参数"属于算法的一部分，跟着算法走最不容易失联。

⚠️ 本模块的算法与重构前的 `posture_monitor.py` **逐字等价**。
   重构只搬位置和拆职责，绝不顺手改判定规则 —— 规则改动必须由真实审核数据说话
   （见 `suggest_threshold` 的注释）。
"""

import math
import statistics

from posture import texts

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

# 侧向机位的偏航角：0°=正对，90°=正侧。纯侧面模式建议 80~90°。
# 只影响侧面指标（用于把被压缩的前后位移还原回矢状面真实量）。
SIDE_YAW_DEG = 90.0

# MediaPipe 33 点索引
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24


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
        issues.append(texts.LOOK_DOWN)
    b = base["head_tilt"]
    if cur["head_tilt"] - b > THRESHOLDS["head_tilt"]:
        issues.append(texts.HEAD_TILT)
    b = base["shoulder"]
    if cur["shoulder"] - b > THRESHOLDS["shoulder"]:
        issues.append(texts.SHOULDER_SLOPE)
    b = base["span"]
    if b > 1e-6 and (cur["span"] - b) / b > THRESHOLDS["too_close"]:
        issues.append(texts.TOO_CLOSE)
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
        issues.append(texts.HEAD_FWD)

    b = base.get("trunk_angle")
    if b is not None and "trunk_angle" in cur and \
            cur["trunk_angle"] - b > SIDE_THRESHOLDS["trunk_angle"]:
        issues.append(texts.SLUMP)

    # 肩-耳距离收缩：低头伏案 / 趴桌。注意方向是「变小」才算异常。
    b = base.get("sh_ear_len")
    if b and b > 1e-6 and (b - cur.get("sh_ear_len", b)) / b > SIDE_THRESHOLDS["head_collapse"]:
        issues.append(texts.LOOK_DOWN)

    return issues


def fuse(front_issues, side_issues):
    """双机位融合判定——这是双机位相对单路的核心价值。

    两路各自看到的证据相互印证时，可以给出置信度更高的结论：
      正面看到「低头」+ 侧面看到「头前倾」 -> 不是单纯的低头，而是「含胸驼背」。
    单路时这个判断做不出来（正面看不到前后关系，侧面看不出左右对称性）。

    纯侧面模式（--side-only）不调用本函数，所有结论都退回单项。
    """
    merged = list(front_issues) + list(side_issues)
    if texts.LOOK_DOWN in merged and texts.HEAD_FWD in merged:
        merged = [i for i in merged if i not in (texts.LOOK_DOWN, texts.HEAD_FWD)]
        merged.insert(0, texts.ROUNDED_BACK)   # 两路互证，升级为更强的结论
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


# ---------------------------------------------------------------- 标定结果的合理性自检

def sanity_rows(tags, bases):
    """标定结果的合理性自检——别跳，能挡掉大量"看着跑起来了其实是垃圾基线"的情况。

    返回 [(标签, 值或None, 判定, 说明)]，判定 ∈ {"ok", "warn", "info"}。
    图形界面直接拿它上色，CLI 用 print_sanity 打印。
    """
    rows = []
    for tag, b in zip(tags, bases):
        if tag == texts.TAG_FRONT:
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


# ---------------------------------------------------------------- 问题名 -> 触发它的指标

# 审核样本只能用来调**它自己触发的那个指标**：
# 家长标"确实歪了"可能是因为塌腰，而那一帧的 neck_angle 可能很低，
# 拿它去压 neck_angle 的阈值就是错的。
ISSUE_METRIC = {
    texts.HEAD_FWD: "neck_angle",
    texts.SLUMP: "trunk_angle",
    texts.LOOK_DOWN: "head_collapse",
}

# 兼容旧代码：重构前这些常量散在 posture_monitor 里，测试与界面按老名字取用。
__all__ = [
    "THRESHOLDS", "SIDE_THRESHOLDS", "SANE", "SIDE_SANE", "SIDE_YAW_DEG",
    "NOSE", "L_EAR", "R_EAR", "L_SH", "R_SH", "L_HIP", "R_HIP",
    "compute_metrics", "build_baseline", "judge",
    "compute_side_metrics", "judge_side", "fuse",
    "side_score", "suggest_threshold", "sanity_rows", "ISSUE_METRIC",
]

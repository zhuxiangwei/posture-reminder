# -*- coding: utf-8 -*-
"""迟滞与提醒间隔控制（`ReminderGate`）。

⚠️ 这段逻辑**必须只有一份实现**。
早期它内联在 CLI 主循环里；加图形界面时如果照着再写一遍，
「连续 4 秒才提醒」这套规则会在两条路径上慢慢漂移（一处改了另一处忘），
而它恰好是最容易毁掉产品的参数（误报 → 孩子三天就把程序关了）。
所以抽成 ReminderGate，CLI 与 GUI 共用。

算法与重构前的 `posture_monitor.py` 逐字等价（只是搬了家）。
"""

SUSTAIN_SEC = 4.0     # 连续超标多久才提醒
CLEAR_SEC = 3.0       # 恢复正常多久才解除
REMIND_GAP_SEC = 90.0  # **同类**问题两次提醒的最小间隔（防同一问题反复叫）
# **任意两次提醒之间**的下限。
# ⚠️ 它必须是全局的、不能按问题分别计时 —— 否则孩子扭一下，
#    问题名在 头前倾 / 塌腰 / 低头 之间轮换，每个问题各自都没到间隔，
#    却会轮流触发，变成每几秒响一次（实测踩过）。
# 需求原话：「三十秒之内如果提示过不正确就不提示，即便不正确的原因也不一样。」
REMIND_GAP_ANY_SEC = 30.0


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

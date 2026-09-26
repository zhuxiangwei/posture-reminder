#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐姿小卫士 · 图形界面（PySide6 / Qt）
=====================================

给**孩子**用的界面。目标：双击就能跑，不需要看懂任何数字和参数。

    python posture_app.py            # 正常启动（自动开始监测）
    python posture_app.py --demo     # 演示模式：不连摄像头，循环展示各种状态
    python posture_app.py --selftest # 只构建界面并退出（离屏，冒烟测试）

为什么用 PySide6 而不是 tkinter
-------------------------------
tkinter 依赖 Python 发行版**自带** tcl/tk。实测本机的 Python 3.13 便携版不含 tkinter，
`import tkinter` 直接 ModuleNotFoundError；而 Qt 是自带二进制的 pip 包，
装完就能用，不受发行版影响。另外 QSS 能做圆角大按钮、平滑字体，更适合儿童界面。

设计原则（都来自报告 §8 的儿童场景研究）
----------------------------------------
1. **零操作启动**：双击即开始。孩子不该需要按任何按钮才能被提醒。
2. **先看脸色，再看字**：状态用大色块 + 手绘表情表达，文字只是补充。
3. **不用红色**：警示用暖橙。红 = 惩罚感，会让孩子想关掉程序。
4. **正向激励**：坐得好时是绿色大字「坐得真棒！」，并把"坐得好的时间""坐姿良好率"
   "连续良好"放在最显眼处，而不是只显示犯了几次错。
5. **设置与孩子隔离**：阈值、标定、提醒节奏全部收进「家长设置」，孩子界面看不到。

技术要点
--------
- 推理跑在 PostureEngine 的独立线程里；Qt 这边只用 QTimer 读快照，绝不跨线程碰控件
- 预览帧用 QImage.Format_BGR888 直接包装 numpy 缓冲，**不需要编解码往返**
- 表情用 QPainter 画，不用 emoji —— emoji 字体在不同 Windows 上差异大，可能变方块
"""

import argparse
import os
import sys
import time

from PySide6.QtCore import Qt, QEvent, QTimer, QRectF, QPointF
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QKeyEvent, QPainter,
                           QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
                               QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSizePolicy, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import posture_monitor as pm              # noqa: E402
import posture_engine as pe               # noqa: E402

# ---------------- 配色（浅色、暖、不用红） ----------------
LEVEL_STYLE = {
    "good":     {"fg": "#1E8A4C", "bg": "#E6F6EC", "face": "#2E9E5B"},
    "warn":     {"fg": "#B4701A", "bg": "#FDF3E1", "face": "#E8A33D"},
    "nodetect": {"fg": "#5C6B80", "bg": "#EDF1F7", "face": "#97A5B8"},
    "idle":     {"fg": "#7A8798", "bg": "#EEF1F6", "face": "#AEB8C6"},
}

QSS = """
QWidget { background: #F4F7FB; color: #25303F; }
QLabel { background: transparent; }
#logo { font-size: 17px; font-weight: 700; }
#statusCard { border-radius: 16px; }
#bigText { font-weight: 700; }
#subText { color: #6B7A8C; }
#card { background: #FFFFFF; border: 1px solid #E1E7EF; border-radius: 14px; }
#cardTitle { color: #7A8798; font-size: 12px; }
#metricName { color: #7A8798; font-size: 13px; }
#metricValue { color: #1E8A4C; font-size: 17px; font-weight: 700; }
#hint { color: #7A8798; font-size: 12px; }
#preview { background: #DCE3EC; color: #7A8798; border-radius: 10px; }
QPushButton#primary, QPushButton#secondary, QPushButton#danger,
QPushButton#save, QPushButton#ghost {
    border: none; border-radius: 12px; padding: 14px 24px; font-size: 14px; font-weight: 700;
}
QPushButton#primary   { background: #5B7A9B; color: #FFFFFF; }
QPushButton#primary:hover   { background: #6C8BA9; }
QPushButton#secondary { background: #7E8FA6; color: #FFFFFF; }
QPushButton#secondary:hover { background: #8E9FB6; }
QPushButton#save      { background: #3C7D55; color: #FFFFFF; }
QPushButton#save:hover { background: #488D62; }
QPushButton#ghost     { background: #E1E7EF; color: #25303F; padding: 10px 16px; }
QPushButton#ghost:hover { background: #D5DDE8; }
QPushButton#link { background: transparent; color: #7A8798; font-size: 12px; border: none; }
QPushButton#link:hover { color: #25303F; }
QProgressBar { border: none; background: #FFFFFF; border-radius: 6px; height: 12px; }
QProgressBar::chunk { background: #7E8FA6; border-radius: 6px; }
QTabWidget::pane { border: 1px solid #E1E7EF; border-radius: 10px; background: #FFFFFF; }
QTabBar::tab { padding: 8px 16px; background: transparent; color: #7A8798; }
QTabBar::tab:selected { color: #25303F; font-weight: 700; }
QLineEdit { border: 1px solid #D5DDE8; border-radius: 8px; padding: 6px 8px;
            background: #FFFFFF; }
QTextEdit { border: 1px solid #E1E7EF; border-radius: 10px; background: #FBFCFE; }
"""


def fmt_dur(sec):
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec} 秒"
    if sec < 3600:
        return f"{sec // 60} 分 {sec % 60} 秒"
    return f"{sec // 3600} 小时 {(sec % 3600) // 60} 分"


def apply_app_font(app):
    """挑一个能正常显示中文的字体。

    ⚠️ 要用 QFontDatabase.families() 查**系统里真实存在**的字体族。
    不要用 QFont(name).exactMatch() —— 字体在真正解析前，
    这个返回值不代表"系统里有没有它"，会误判。
    """
    from PySide6.QtGui import QFontDatabase
    fams = set(QFontDatabase.families())
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑", "SimHei"):
        if name in fams:
            app.setFont(QFont(name, 10))
            return name
    app.setFont(QFont("", 10))
    return None


# ---------------------------------------------------------------- 手绘表情

class FaceWidget(QWidget):
    """手绘表情。Canvas 式绘制，不依赖 emoji 字体。"""

    def __init__(self, size=116):
        super().__init__()
        self._size = size
        self.setFixedSize(size, size)
        self._level = "idle"
        self._color = QColor(LEVEL_STYLE["idle"]["face"])

    def set_level(self, level):
        col = QColor(LEVEL_STYLE.get(level, LEVEL_STYLE["idle"])["face"])
        if level != self._level or col != self._color:
            self._level = level
            self._color = col
            self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        s = float(self._size)
        c = s / 2.0
        r = s * 0.40

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._color))
        p.drawEllipse(QPointF(c, c), r, r)

        # 眼睛
        p.setBrush(QBrush(QColor("#FFFFFF")))
        eye_y = c - r * 0.28
        ew = r * 0.13
        ex = r * 0.38
        for dx in (-ex, ex):
            p.drawEllipse(QPointF(c + dx, eye_y), ew, ew * 1.35)

        # 嘴：good=上扬；warn=轻微下压（不是哭脸，别做成惩罚）；其它=平
        pen = QPen(QColor("#FFFFFF"))
        pen.setWidthF(max(2.0, s * 0.035))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        mw = r * 0.52
        my = c + r * 0.30
        if self._level == "good":
            p.drawArc(QRectF(c - mw, my - r * 0.62, mw * 2, r * 0.96), 200 * 16, 140 * 16)
        elif self._level == "warn":
            p.drawArc(QRectF(c - mw, my - r * 0.10, mw * 2, r * 0.96), 20 * 16, 140 * 16)
        else:
            p.drawLine(QPointF(c - mw * 0.8, my + r * 0.18), QPointF(c + mw * 0.8, my + r * 0.18))
        p.end()


# ---------------------------------------------------------------- 主窗口

class PostureApp(QWidget):
    POLL_MS = 150

    def __init__(self, demo=False, autostart=True, log_path=None, log_enabled=True):
        super().__init__()
        self.demo = demo
        self.engine = None
        self._demo_speaker = None
        self._log_lines = []
        # 运行日志同时落盘：GUI 里那份只活在内存（关掉就没了），
        # 出了问题只能靠人复述。落盘后把 logs\run-*.log 发过来即可查。
        self.runlog = pe.RunLog(log_path, enabled=log_enabled)
        self._demo_stats = pe.PostureEngine._blank_stats()
        self._demo_i = 0

        self.setWindowTitle("坐姿小卫士" + ("（演示）" if demo else ""))
        w, h = self._fit_to_screen(900, 620)
        self.resize(w, h)
        self.setMinimumSize(700, 500)

        self._build_ui()

        # ⚠️ 声音按钮的初始文字必须读配置，不能写死"声音：开"。
        #    否则存过静音之后，按钮显示的状态是错的（显示"开"其实静音着）。
        cfg0 = pe.load_config()
        self.btn_sound.setText("声音：关" if cfg0.get("mute") else "声音：开")

        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        if demo:
            self._setup_demo()
            self._demo_timer = QTimer(self)
            self._demo_timer.setInterval(1500)
            self._demo_timer.timeout.connect(self.demo_step)
            self._demo_timer.start()
        elif autostart:
            QTimer.singleShot(300, self._start_engine)

    # ---------------- 构建界面

    def _fit_to_screen(self, want_w, want_h):
        """按屏幕**可用区域**收敛初始尺寸，并居中。

        ⚠️ HDPI 下 Qt 用的是**逻辑像素**：1920x1080 开 150% 缩放时，
           桌面逻辑尺寸只有 1280x720，任务栏再占掉一条 —— 一个 720 高的窗口
           根本放不下（会出现标题栏被顶出屏幕、底部按钮够不到）。
           所以初始尺寸绝不能写死，必须按 availableGeometry() 收敛。
        """
        scr = self.screen() or QApplication.primaryScreen()
        if scr is None:
            return want_w, want_h
        av = scr.availableGeometry()
        w = max(min(700, av.width()), min(want_w, int(av.width() * 0.94)))
        h = max(min(500, av.height()), min(want_h, int(av.height() * 0.94)))
        self.move(av.x() + max(0, (av.width() - w) // 2),
                  av.y() + max(0, (av.height() - h) // 3))
        return w, h

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 14)
        root.setSpacing(12)

        # ---- 标题栏 ----
        head = QHBoxLayout()
        logo = QLabel("坐姿小卫士")
        logo.setObjectName("logo")
        head.addWidget(logo)
        head.addStretch(1)
        btn_parent = QPushButton("家长设置")
        # 不可获得焦点：否则焦点落上来后空格会去点它，抢掉「暂停/继续」的快捷键
        btn_parent.setFocusPolicy(Qt.NoFocus)
        btn_parent.setObjectName("link")
        btn_parent.setCursor(Qt.PointingHandCursor)
        btn_parent.clicked.connect(self._open_parent)
        head.addWidget(btn_parent)
        root.addLayout(head)

        # ---- 状态卡 ----
        self.status_card = QFrame()
        self.status_card.setObjectName("statusCard")
        sc = QHBoxLayout(self.status_card)
        sc.setContentsMargins(26, 20, 26, 20)
        sc.setSpacing(24)

        self.face = FaceWidget(96)
        sc.addWidget(self.face, 0, Qt.AlignVCenter)

        col = QVBoxLayout()
        col.setSpacing(6)
        self.lbl_big = QLabel(pe.TEXT_IDLE)
        self.lbl_big.setObjectName("bigText")
        self.lbl_big.setFont(QFont(self.lbl_big.font().family(), 22, QFont.Bold))
        self.lbl_big.setWordWrap(True)
        col.addWidget(self.lbl_big)

        self.lbl_sub = QLabel("")
        self.lbl_sub.setObjectName("subText")
        self.lbl_sub.setWordWrap(True)
        col.addWidget(self.lbl_sub)
        col.addStretch(1)
        sc.addLayout(col, 1)
        root.addWidget(self.status_card)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(12)
        self.bar.hide()
        root.addWidget(self.bar)

        # ---- 中部：预览 + 今日表现 ----
        mid = QHBoxLayout()
        mid.setSpacing(12)

        left = QFrame()
        left.setObjectName("card")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(14, 12, 14, 14)
        t1 = QLabel("摄像头画面")
        t1.setObjectName("cardTitle")
        lv.addWidget(t1)
        self.preview = QLabel("等待画面…")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(340, 190)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lv.addWidget(self.preview, 1)
        mid.addWidget(left, 3)

        right = QFrame()
        right.setObjectName("card")
        right.setFixedWidth(300)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 12, 16, 14)
        t2 = QLabel("今天的表现")
        t2.setObjectName("cardTitle")
        rv.addWidget(t2)
        grid = QGridLayout()
        grid.setVerticalSpacing(12)
        self.metrics = {}
        for i, (key, label) in enumerate((("good", "坐得好的时间"), ("ratio", "坐姿良好率"),
                                          ("streak", "连续良好"), ("reminders", "提醒次数"))):
            n = QLabel(label)
            n.setObjectName("metricName")
            v = QLabel("—")
            v.setObjectName("metricValue")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(n, i, 0)
            grid.addWidget(v, i, 1)
            self.metrics[key] = v
        grid.setColumnStretch(1, 1)
        rv.addLayout(grid)
        rv.addStretch(1)
        mid.addWidget(right, 0)
        root.addLayout(mid, 1)

        # ---- 底部按钮 ----
        foot = QHBoxLayout()
        foot.setSpacing(10)
        self.btn_toggle = QPushButton("暂停")
        self.btn_toggle.setObjectName("primary")
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.clicked.connect(self._on_toggle)
        self.btn_toggle.setToolTip("暂停 / 继续（快捷键：空格键）")
        foot.addWidget(self.btn_toggle)

        self.btn_sound = QPushButton("声音：开")
        self.btn_sound.setObjectName("secondary")
        self.btn_sound.setCursor(Qt.PointingHandCursor)
        self.btn_sound.clicked.connect(self._on_sound)
        foot.addWidget(self.btn_sound)

        foot.addStretch(1)
        self.btn_recal = QPushButton("重新学习坐姿")
        self.btn_recal.setObjectName("secondary")
        self.btn_recal.setCursor(Qt.PointingHandCursor)
        self.btn_recal.clicked.connect(self._on_recal)
        foot.addWidget(self.btn_recal)
        root.addLayout(foot)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setObjectName("hint")
        self.lbl_hint.setWordWrap(True)

        # ---- 空格键：暂停 / 继续 ----
        # 用 keyPressEvent（见下面）而不是 QShortcut，原因：
        #   QShortcut 要走 Qt 的快捷映射、依赖窗口处于激活状态，
        #   实测在离屏环境下**一次都不触发**（连直接给窗口发按键也不触发），
        #   行为不可靠、也没法自动化验证。keyPressEvent 是纯事件分发，确定、可测。
        #
        # ⚠️ 但有个前提：**按钮不能抢走空格**。Qt 默认把空格当作"点击"当前
        #    焦点按钮，焦点一旦落在按钮上，空格就去点那个按钮、到不了窗口。
        #    所以把底部按钮设成不可获得焦点。这个程序是鼠标 / 触屏操作的，
        #    不需要 Tab 在按钮间跳，代价可接受。
        for b in (self.btn_toggle, self.btn_sound, self.btn_recal):
            b.setFocusPolicy(Qt.NoFocus)
        root.addWidget(self.lbl_hint)

        self._apply_level("idle", pe.TEXT_IDLE)

    # ---------------- 渲染

    def _apply_level(self, level, text, sub="", progress=None):
        st = LEVEL_STYLE.get(level, LEVEL_STYLE["idle"])
        self.status_card.setStyleSheet(
            f"#statusCard {{ background: {st['bg']}; border-radius: 16px; }}")
        self.lbl_big.setStyleSheet(
            f"#bigText {{ color: {st['fg']}; font-weight: 700; }}")
        self.lbl_sub.setStyleSheet(f"#subText {{ color: #6B7A8C; }}")
        self.lbl_big.setText(text)
        self.lbl_sub.setText(sub)
        self.face.set_level(level)

        if progress is None:
            self.bar.hide()
        else:
            self.bar.show()
            self.bar.setValue(int(max(0, min(100, progress * 100))))

    def _render_preview(self, frame, kps=None):
        if frame is None or pm.cv2 is None:
            return
        try:
            img = frame
            if kps is not None:
                img = frame.copy()
                h, w = img.shape[:2]
                for i in pm.KEYPOINTS_DRAW:
                    pm.cv2.circle(img, (int(kps[i].x * w), int(kps[i].y * h)), 3,
                                  (0, 220, 0), -1)
            if not img.flags["C_CONTIGUOUS"]:
                img = img.copy()
            h, w, ch = img.shape
            # QImage 不持有 numpy 缓冲，必须 copy() 后再交给 QPixmap
            qimg = QImage(img.data, w, h, ch * w, QImage.Format_BGR888).copy()
            pix = QPixmap.fromImage(qimg)
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation))
            self.preview.setText("")
        except Exception:
            pass      # 预览失败绝不影响提醒主功能

    def _log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._log_lines.append(line)
        del self._log_lines[:-300]
        print(line)
        self.runlog.write(line)         # 落盘失败会被吞掉，不影响主流程
        for w in self.findChildren(QTextEdit):
            if w.objectName() == "logwin":
                w.append(line)

    # ---------------- 控制

    def _start_engine(self):
        if self.engine:
            return
        self.engine = pe.PostureEngine(cfg=pe.load_config(), on_log=self._log)
        self.engine.start()
        self.btn_toggle.setText("暂停")
        self._log("引擎已启动（空格键可暂停 / 继续）")

    def keyPressEvent(self, e):
        """空格键 = 暂停 / 继续。

        ⚠️ `isAutoRepeat()` 必须判：按住空格不放会连续产生按键事件，
           不判的话会在暂停/继续之间疯狂翻转。
        """
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._on_toggle()
            e.accept()
            return
        super().keyPressEvent(e)

    def _on_toggle(self):
        if self.demo:
            return
        if not self.engine:
            self._start_engine()
            return
        phase = self.engine.snapshot()["phase"]
        if phase == "error":
            self._restart()
            return
        if phase == "stopped":
            self.engine = None
            self._start_engine()
            return
        if self.btn_toggle.text() == "暂停":
            self.engine.pause()
            self.btn_toggle.setText("继续")
            self._log("已暂停")
        else:
            self.engine.resume()
            self.btn_toggle.setText("暂停")
            self._log("已继续")

    def _on_sound(self):
        turn_off = self.btn_sound.text() == "声音：开"
        if self.engine:
            self.engine.set_audio(mute=turn_off)
        if self._demo_speaker:
            self._demo_speaker.mute = turn_off
        cfg = pe.load_config()
        cfg["mute"] = turn_off
        pe.save_config(cfg)
        self.btn_sound.setText("声音：关" if turn_off else "声音：开")
        self._log("声音已" + ("关闭" if turn_off else "开启"))
        if not turn_off and self._demo_speaker:
            self._demo_speaker.chime("done")     # 开声音时给个即时反馈

    def _on_recal(self):
        if self.demo or not self.engine:
            return
        ok = QMessageBox.question(
            self, "重新学习坐姿",
            "请让孩子在座位上端坐好，然后点「确定」。\n\n"
            "接下来约 10 秒会重新采集基线。机位没动过就不需要做这一步。",
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if ok != QMessageBox.Ok:
            return
        self.engine.recalibrate()
        self.btn_toggle.setText("暂停")
        self._log("已请求重新标定")

    def _restart(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self._start_engine()

    def closeEvent(self, ev):
        if self.engine:
            self.engine.stop()
        self._timer.stop()
        super().closeEvent(ev)

    # ---------------- 轮询

    def _tick(self):
        if self.engine:
            self._render_snapshot(self.engine.snapshot())

    def _render_snapshot(self, s):
        phase = s.get("phase", "idle")
        self._render_preview(s.get("preview"), s.get("kps"))

        if phase == "error":
            self._apply_level("warn", "有点小问题", s.get("error") or "")
            self.btn_toggle.setText("重试")
            self.lbl_hint.setText(
                "修好后点「重试」。最常见的两个原因：摄像头被别的程序占用（会议软件／"
                "相机 App），或画面里始终看不到人导致标定失败。")
            return

        if phase == "opening":
            self._apply_level("idle", "正在打开摄像头…", "第一次启动会慢几秒")
            return

        if phase == "calibrating":
            c = s.get("calib") or {}
            done, need = int(c.get("done", 0)), max(1, int(c.get("need", 15)))
            self._apply_level(
                "idle", "请坐正，正在学习你的坐姿…",
                f"保持住，还有 {max(0.0, c.get('sec_left', 0)):.0f} 秒　"
                f"（已采集 {done}/{need}）",
                progress=done / need)
            self.lbl_hint.setText(
                "这一步看着像「没反应」，其实正在采集基线。"
                "请让孩子的头和肩膀都出现在画面里。")
            return

        if phase in ("idle", "stopped"):
            self._apply_level("idle", pe.TEXT_IDLE)
            return

        if self.btn_toggle.text() == "继续":
            self._apply_level("idle", pe.TEXT_PAUSED, "点「继续」接着看")
            return

        level = s.get("level", "idle")
        sub = ""
        pend = s.get("pending")
        if pend:
            need = float(self.engine.cfg.get("sustain_sec", pm.SUSTAIN_SEC)) if self.engine \
                else pm.SUSTAIN_SEC
            sub = f"再保持 {max(0.0, need - pend[1]):.0f} 秒，会自动提醒一下"
        if level == "nodetect":
            sub = "请坐到座位中间，让摄像头看得到你"
        self._apply_level(level, s.get("text") or pe.TEXT_IDLE, sub)

        st = s.get("stats") or pe.PostureEngine._blank_stats()
        self.metrics["good"].setText(fmt_dur(st.get("good_sec", 0)))
        r = st.get("good_ratio")
        self.metrics["ratio"].setText("—" if r is None else f"{r * 100:.0f}%")
        self.metrics["streak"].setText(fmt_dur(st.get("streak_sec", 0)))
        self.metrics["reminders"].setText(str(st.get("reminders", 0)))

        hint = "已读回上次的基线，这次不需要重新学习。" if s.get("baseline") else ""
        rate = s.get("detect_rate")
        if rate is not None:
            if rate < 0.3:
                hint = (f"⚠️ 只有 {rate * 100:.0f}% 的时间看得到人——请检查机位、距离和光线，"
                        "否则判定不准。")
            elif not hint:
                hint = f"画面里能看到人的比例：{rate * 100:.0f}%"
        self.lbl_hint.setText(hint)

    # ---------------- 家长设置

    def _open_parent(self):
        dlg = ParentDialog(self)
        dlg.exec()

    # ---------------- 演示模式

    def _setup_demo(self):
        self.setWindowTitle("坐姿小卫士（演示）")
        self.btn_toggle.setText("演示中")
        self.btn_toggle.setEnabled(False)
        self.btn_recal.setEnabled(False)

        # 演示模式也要能出声 —— 否则「试听」这件事没法做，
        # 而报告 §7 明确要求"让孩子听一遍，确认不刺耳、不反感"。
        cfg = pe.load_config()
        self._demo_speaker = pe.Speaker(mute=bool(cfg.get("mute")),
                                        volume=cfg.get("volume", "中"))
        mute = "静音中" if cfg.get("mute") else f"开（音量档位：{cfg.get('volume', '中')}）"
        self._log(f"演示模式：不连摄像头，循环展示各种状态；声音 {mute}")
        self.lbl_hint.setText(
            "演示模式 —— 正在循环展示「坐得好／提醒／看不到人」三种状态，"
            f"语音提醒也会播。当前声音：{mute}（点下面「声音」按钮可切换）。"
            "去掉 --demo 参数即可正常使用。")

    def demo_step(self):
        # (状态, 大字, 副标题, 对应的问题名) —— 问题名用来取 voice/<名字>.wav
        seq = [("good", pe.TEXT_GOOD, "", None),
               ("good", pe.TEXT_GOOD, "", None),
               ("warn", "头抬高一点点～", "再保持 2 秒，会自动提醒一下", "头前倾"),
               ("warn", "背挺直一点～", "再保持 1 秒，会自动提醒一下", "塌腰"),
               ("nodetect", pe.TEXT_NODETECT, "请坐到座位中间，让摄像头看得到你", None)]
        st = self._demo_stats
        dt = 1.5
        level, text, sub, issue = seq[self._demo_i % len(seq)]
        self._demo_i += 1

        st["total_sec"] += dt
        if level == "good":
            st["good_sec"] += dt
            st["streak_sec"] += dt
            st["best_streak_sec"] = max(st["best_streak_sec"], st["streak_sec"])
        elif level == "warn":
            st["streak_sec"] = 0.0
            st["reminders"] += 1
        else:
            st["streak_sec"] = 0.0
            st["total_sec"] = max(0.0, st["total_sec"] - dt)   # 没人在画面里不计入
        st["good_ratio"] = (st["good_sec"] / st["total_sec"]) if st["total_sec"] > 0 else None

        self._apply_level(level, text, sub)
        self._render_preview(self._demo_frame())
        self.metrics["good"].setText(fmt_dur(st["good_sec"]))
        self.metrics["ratio"].setText(
            "—" if st["good_ratio"] is None else f"{st['good_ratio'] * 100:.0f}%")
        self.metrics["streak"].setText(fmt_dur(st["streak_sec"]))
        self.metrics["reminders"].setText(str(st["reminders"]))

        # 语音：真机上该响的时候这里也要响，否则演示模式预览不到真实体验
        # ⚠️ 坐正了**刻意不发声**（需求：坐姿没问题就不要发出任何声音），
        #    所以这里只有 warn 会播，good 不播。
        if self._demo_speaker and level == "warn" and issue:
            queued = self._demo_speaker.say(issue)
            self._log(f"[演示] 提醒「{issue}」"
                      f"（{'已排入播放队列' if queued else '静音中，未播放'}）")

    def _demo_frame(self):
        """造一张假的摄像头画面，把预览链路也跑到。"""
        try:
            import numpy as np
            f = np.full((270, 480, 3), 226, dtype=np.uint8)
            pm.cv2.rectangle(f, (150, 62), (330, 250), (206, 214, 224), -1)
            pm.cv2.circle(f, (240, 110), 42, (188, 198, 210), -1)
            pm.cv2.putText(f, "DEMO", (16, 250), pm.cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (150, 160, 175), 2)
            return f
        except Exception:
            return None


# ---------------------------------------------------------------- 审核判定面板

class ReviewPanel(QWidget):
    """让家长审核程序判过的现场。

    这是「阈值标定」的唯一现实路径：
      原计划是录 15~20 分钟视频再逐帧人工标注 —— 家长不会做，数据收不上来，
      而"误报率 < 10%"这个验收指标就没有任何办法算出来。
      改成"提醒响了自动存一张、家长顺手点一下"，成本几乎为零。

    ⚠️ 界面必须说清能力边界：**只能发现误报，发现不了漏报**
       （漏报时程序压根没存图）。所以引擎另外按固定间隔存了"判为良好"的抽样。
    """

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.store = self._store()
        self._items = []
        self._idx = 0
        self._build()
        self.reload()

    def _store(self):
        eng = getattr(self.app, "engine", None)
        if eng is not None:
            return eng.review
        cfg = pe.load_config()      # 演示模式/引擎未起时，也能看已存的
        return pe.ReviewStore(max_files=cfg.get("review_max_files", 300), enabled=True)

    # ---------------- 构建

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        head = QLabel(
            "程序判过的现场都存在这里。<b>你只需要点一下「对／不对」</b>——"
            "这批标注是标定阈值的唯一数据来源，没有它误报率永远算不出来。\n"
            "⚠️ 这里<b>只能发现「误报」</b>（判歪了其实没歪）；"
            "<b>发现不了「漏报」</b>，因为漏报时程序根本不会存图 —— "
            "所以另外按固定间隔存了「判为良好」的抽样，用来查漏报。\n"
            "🔒 图片只存在本机 review 目录，不上传任何地方；可一键清空。")
        head.setWordWrap(True)
        head.setTextFormat(Qt.RichText)
        v.addWidget(head)

        # ---- 本功能的开关（放在同一页，免得家长去别处找）----
        cfg = pe.load_config()
        row0 = QHBoxLayout()
        self.chk_on = QCheckBox("留存现场图供审核（仅存本机）")
        self.chk_on.setChecked(bool(cfg.get("review_enabled", True)))
        row0.addWidget(self.chk_on)
        row0.addWidget(QLabel("判为良好时，每"))
        self.ed_interval = QLineEdit(str(int(float(cfg.get("review_sample_sec", 180) or 0))))
        self.ed_interval.setFixedWidth(64)
        row0.addWidget(self.ed_interval)
        row0.addWidget(QLabel("秒另存一张抽样"))
        b = QPushButton("保存这些设置")
        b.setObjectName("ghost")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self._save_settings)
        row0.addWidget(b)
        row0.addStretch(1)
        v.addLayout(row0)

        self.chk_full = QCheckBox("提醒现场另存一张原图"
                                  "（只为保留将来模型微调的可能，约 200KB/张）")
        self.chk_full.setChecked(bool(cfg.get("review_keep_fullres", True)))
        v.addWidget(self.chk_full)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("font-weight:700;")
        v.addWidget(self.lbl_stats)

        self.img = QLabel("（没有待审核的现场）")
        self.img.setObjectName("preview")
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setMinimumHeight(200)
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.img, 1)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        v.addWidget(self.info)

        row = QHBoxLayout()
        self.btn_notrigger = QPushButton("其实坐得好（误报）")
        self.btn_trigger = QPushButton("确实歪了（判对了）")
        for b, obj in ((self.btn_notrigger, "secondary"), (self.btn_trigger, "save")):
            b.setObjectName(obj)
            b.setCursor(Qt.PointingHandCursor)
            row.addWidget(b)
        self.btn_notrigger.clicked.connect(lambda: self._label(False))
        self.btn_trigger.clicked.connect(lambda: self._label(True))
        self.btn_skip = QPushButton("跳过")
        self.btn_skip.setObjectName("ghost")
        self.btn_skip.setCursor(Qt.PointingHandCursor)
        self.btn_skip.clicked.connect(self._skip)
        row.addWidget(self.btn_skip)
        row.addStretch(1)
        self.btn_clear = QPushButton("清空全部")
        self.btn_clear.setObjectName("ghost")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self._clear)
        row.addWidget(self.btn_clear)
        v.addLayout(row)

        t = QLabel("阈值建议（用你已审核的样本反推；只作参考，改不改由你决定）")
        t.setObjectName("cardTitle")
        v.addWidget(t)
        self.suggest = QTextEdit()
        self.suggest.setReadOnly(True)
        self.suggest.setFixedHeight(120)
        v.addWidget(self.suggest)

    # ---------------- 渲染

    def reload(self):
        self._items = self.store.items("pending")
        self._idx = 0
        self._render()

    def _render(self):
        st = self.store.stats()
        fp = "—" if st["fp_rate"] is None else f"{st['fp_rate']*100:.0f}%"
        self.lbl_stats.setText(
            f"待审核 {st['pending']} 条　｜　已审核 {st['labeled']} 条"
            f"（你标为误报 {st['wrong']} 条、漏报 {st['missed']} 条）　｜　"
            f"提醒的误报率 {fp}")

        if not self._items:
            self.img.setPixmap(QPixmap())
            self.img.setText("（没有待审核的现场）")
            self.info.setText("")
            for b in (self.btn_notrigger, self.btn_trigger, self.btn_skip):
                b.setEnabled(False)
            self.btn_notrigger.setText("其实坐得好（误报）")
            self.btn_trigger.setText("确实歪了（判对了）")
        else:
            for b in (self.btn_notrigger, self.btn_trigger, self.btn_skip):
                b.setEnabled(True)
            it = self._items[self._idx % len(self._items)]
            d = it["data"]
            try:
                pix = QPixmap(it["jpg"])
                self.img.setText("")
                self.img.setPixmap(pix.scaled(self.img.size(), Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))
            except Exception:
                self.img.setText("（图片读取失败）")

            reminder = d.get("kind") == "reminder"
            if reminder:
                self.btn_notrigger.setText("其实坐得好（误报）")
                self.btn_trigger.setText("确实歪了（判对了）")
                title = f"当时判断：<b>{d.get('issue')}</b>（程序认为歪了，已提醒）"
            else:
                self.btn_notrigger.setText("确实坐得好（判对了）")
                self.btn_trigger.setText("其实歪了（漏报！）")
                title = "当时判断：<b>坐姿良好</b>（程序没提醒）"

            m, b = d.get("metrics") or {}, d.get("baseline") or {}
            lines = [title, f"时间：{d.get('ts')}　（第 {self._idx % len(self._items) + 1}"
                            f" / {len(self._items)} 条）"]
            for key, name, unit in (("neck_angle", "颈部前倾角", "°"),
                                    ("trunk_angle", "躯干倾角", "°"),
                                    ("sh_ear_len", "肩-耳距离", "px")):
                cur, base = m.get(key), b.get(key)
                if cur is None:
                    lines.append(f"{name}：拿不到")
                elif base is None:
                    lines.append(f"{name}：{cur:.1f}{unit}")
                else:
                    d_ = cur - base
                    lines.append(f"{name}：{cur:.1f}{unit}　（基线 {base:.1f}，差 {d_:+.1f}）")
            self.info.setText("<br>".join(lines))

        self._render_suggest()

    def _render_suggest(self):
        try:
            cfg = pe.load_config()
            res = self.store.suggest_all(cfg)
        except Exception as e:
            self.suggest.setPlainText(f"（算建议值时出错：{e}）")
            return
        if not res:
            self.suggest.setPlainText(
                "样本还不够，暂时给不出建议。\n"
                "至少需要同一指标的「判对」和「误报」样本各 1 条 —— 多审核几十条就有了。")
            return
        names = {"neck_angle": "颈部前倾角（头前倾）",
                 "trunk_angle": "躯干倾角（塌腰）",
                 "head_collapse": "肩耳收缩（低头）"}
        out = []
        for metric, r in sorted(res.items()):
            cur = "—" if r["current"] is None else f"{r['current']:.1f}"
            out.append(f"· {names.get(metric, metric)}：当前阈值 {cur} → "
                       f"建议 {r['suggested']:.1f}")
            detail = (f"    样本 {r['n_should'] + r['n_shouldnt']} 条"
                      f"（该报警 {r['n_should']} / 误报 {r['n_shouldnt']}）；"
                      f"改后误报 {r['fp_at_suggested']}、漏报 {r['fn_at_suggested']}")
            if r["fp_at_current"] is not None:
                detail += f"（当前阈值下误报 {r['fp_at_current']}）"
            out.append(detail)
        out.append("")
        out.append("提示：误报比漏报致命，所以建议值偏向「宁可少叫」。")
        out.append("改法：到「判定阈值」页填写，或直接改 config.json。")
        self.suggest.setPlainText("\n".join(out))

    # ---------------- 操作

    def _save_settings(self):
        try:
            sec = float(self.ed_interval.text().strip())
            if sec < 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "填错了", "抽样间隔要填数字（秒），且不能为负。")
            return
        cfg = pe.load_config()
        cfg["review_enabled"] = bool(self.chk_on.isChecked())
        cfg["review_sample_sec"] = sec
        cfg["review_keep_fullres"] = bool(self.chk_full.isChecked())
        if not pe.save_config(cfg):
            QMessageBox.critical(self, "保存失败", "写不进 config.json。")
            return
        eng = getattr(self.app, "engine", None)
        if eng is not None:
            eng.cfg.update(cfg)
            eng.review.enabled = cfg["review_enabled"]
        if hasattr(self.app, "_log"):
            self.app._log(f"[审核] 留存={cfg['review_enabled']} "
                          f"抽样间隔={sec:.0f}s 存原图={cfg['review_keep_fullres']}")
        QMessageBox.information(self, "已保存",
                                "设置已保存并即时生效（抽样间隔从下一次抽样开始按新值）。")

    def _label(self, should_trigger):
        if not self._items:
            return
        it = self._items[self._idx % len(self._items)]
        self.store.label(it, should_trigger)
        if hasattr(self.app, "_log"):
            self.app._log(f"[审核] {it['data'].get('ts')} "
                          f"{it['data'].get('issue') or '良好抽样'} -> "
                          f"{'确实该报警' if should_trigger else '不该报警'}")
        self._items.pop(self._idx % len(self._items))
        if self._idx >= len(self._items):
            self._idx = 0
        self._render()

    def _skip(self):
        if not self._items:
            return
        self._idx = (self._idx + 1) % len(self._items)
        self._render()

    def _clear(self):
        ok = QMessageBox.question(
            self, "清空全部现场图",
            "会删除 review 目录下所有已存和已审核的现场图，不可恢复。\n\n确定？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok == QMessageBox.Yes:
            n = self.store.clear_all()
            if hasattr(self.app, "_log"):
                self.app._log(f"[审核] 已清空 {n} 个文件")
            self.reload()


# ---------------------------------------------------------------- 家长设置

class ParentDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("家长设置")
        # 同样不能写死高度：HDPI 下逻辑分辨率可能只有 720 上下
        scr = self.screen() or QApplication.primaryScreen()
        if scr is not None:
            av = scr.availableGeometry()
            self.resize(min(640, int(av.width() * 0.90)), min(620, int(av.height() * 0.90)))
        else:
            self.resize(640, 620)
        cfg = pe.load_config()
        self._cfg = cfg
        self._inputs = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)

        t = QLabel("家长设置")
        t.setObjectName("logo")
        root.addWidget(t)
        s = QLabel("这些参数孩子不需要看到。改完点「保存并应用」。")
        s.setObjectName("hint")
        root.addWidget(s)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        # --- 页1 机位与节奏 ---
        p1 = self._tab_scroll(tabs, "机位与节奏")
        # ⚠️ 不能写 QFormLayout(p1) —— 用 p1 当父控件会**顶替**该页已有的布局
        #    （Qt 报 "QWidget which already has a layout"，且页面排版会被打乱）。
        #    必须先无父构造，再加进该页的布局里。
        f1 = QFormLayout()
        p1.layout().addLayout(f1)
        f1.setSpacing(10)
        for key, label, kind, hint in (
                ("camera", "摄像头索引", int, "本机只接了一颗 USB 摄像头，填 0；接多颗时用 --list-cameras 查"),
                ("side_yaw", "机位偏航角（度）", float, "0=正对，90=正侧。纯侧面建议 80~90，别用 45"),
                ("calib_sec", "标定时长（秒）", float, "孩子端坐保持多久来采集基线"),
                ("fps_sample", "采样帧率", int, "5 就够。坐姿是秒级慢过程，调高只白耗电")):
            self._add_row(f1, key, label, cfg.get(key), kind, hint)

        # --- 页2 阈值 ---
        p2 = self._tab_scroll(tabs, "判定阈值")
        warn2 = QLabel("⚠️ 这些是未标定的初始值。必须录真实作业视频回归，否则误报率不可控"
                       "——误报比漏报致命。")
        warn2.setWordWrap(True)
        warn2.setStyleSheet("color:#B4701A;")
        p2.layout().addWidget(warn2)
        f2 = QFormLayout()
        p2.layout().addLayout(f2)
        for key, label, hint in (
                ("neck_angle", "头前倾阈值（度）", "相对基线增加多少度算头前倾"),
                ("trunk_angle", "塌腰阈值（度）", "髋部可见时才生效"),
                ("head_collapse", "低头阈值（比例）", "肩-耳距离收缩多少算低头／趴桌")):
            self._add_row(f2, key, label, cfg["side_thresholds"].get(key), float, hint,
                          box=self._inputs.setdefault("side_thresholds", {}))

        # --- 页3 提醒与声音 ---
        p3 = self._tab_scroll(tabs, "提醒与声音")
        warn3 = QLabel("⚠️ 下面三项是防「抖一下就叫」的关键，不要为了更灵敏调小。"
                       "叫得太勤，孩子三天就把程序关了。")
        warn3.setWordWrap(True)
        warn3.setStyleSheet("color:#B4701A;")
        p3.layout().addWidget(warn3)
        f3 = QFormLayout()
        p3.layout().addLayout(f3)
        for key, label, hint in (
                ("sustain_sec", "连续超标多久才提醒（秒）", "越短越灵敏也越吵"),
                ("clear_sec", "恢复正常多久才解除（秒）", "防止状态来回闪"),
                ("remind_gap_sec", "【同类】问题最小间隔（秒）",
                 "同一个问题两次提醒至少隔这么久"),
                ("remind_gap_any_sec", "【任意】两次提醒的最小间隔（秒）",
                 "跨问题也生效：30 秒内已经提醒过，就一律不再提醒，"
                 "哪怕这次的原因和上次不一样")):
            self._add_row(f3, key, label, cfg.get(key), float, hint)

        self.chk_mute = QCheckBox("静音（只显示，不出声）")
        self.chk_mute.setChecked(bool(cfg.get("mute")))
        p3.layout().addWidget(self.chk_mute)

        vol = QHBoxLayout()
        vol.addWidget(QLabel("音量档位："))
        self.radios = {}
        for lv in ("低", "中", "高"):
            r = QRadioButton(lv)
            r.setChecked(cfg.get("volume", "中") == lv)
            self.radios[lv] = r
            vol.addWidget(r)
        vol.addStretch(1)
        p3.layout().addLayout(vol)

        # 试听 —— 报告 §7 要求"让孩子听一遍，确认不刺耳、不反感"，之前没有任何入口
        voices = self._available_voices()
        if voices:
            t = QLabel("试听（用上面选的音量档位播一遍；试听会忽略静音开关）")
            t.setObjectName("hint")
            t.setWordWrap(True)
            p3.layout().addWidget(t)
            row = QHBoxLayout()
            for name in voices:
                b = QPushButton(name)
                b.setObjectName("ghost")
                b.setCursor(Qt.PointingHandCursor)
                b.clicked.connect(lambda _=False, n=name: self._preview_voice(n))
                row.addWidget(b)
            row.addStretch(1)
            p3.layout().addLayout(row)
            self.lbl_preview = QLabel("")
            self.lbl_preview.setObjectName("hint")
            p3.layout().addWidget(self.lbl_preview)
        else:
            t = QLabel("voice 目录下没有语音文件，程序会退回系统提示音。"
                       "用 tools/make_voice.py 生成，或自己放 wav 进去。")
            t.setObjectName("hint")
            t.setWordWrap(True)
            p3.layout().addWidget(t)
        p3.layout().addStretch(1)

        # --- 页4 机位检查（摆好机位后用它确认，别靠"看着像在跑"）---
        p4 = self._tab_scroll(tabs, "机位检查")
        info = QLabel(
            "怎么用：点下面的「开始观察」，<b>让孩子在座位上往前伸一下头再坐回去</b>，"
            "看「颈部前倾角摆幅」涨到多少。\n\n"
            "⚠️ 如果摆幅很小（<5°），说明摄像头太正对孩子了 —— "
            "正面视角下同侧的耳和肩几乎在一条竖直线上，前后关系被压扁，"
            "程序<b>不会报错，而是静默失效</b>。把摄像头转向孩子侧面，机位摆到 80~90° 正侧。\n\n"
            "判据用「摆幅」而不是「和基线的差值」，所以**不用等标定完成**就能验机位。")
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        p4.layout().addWidget(info)

        self.live = {}
        for key, label in (("neck_angle", "当前颈部前倾角"),
                           ("trunk_angle", "当前躯干倾角"),
                           ("sh_ear_len", "当前肩-耳距离")):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 3, 0, 3)
            n = QLabel(label)
            rl.addWidget(n, 1)
            v = QLabel("—")
            v.setObjectName("metricValue")
            rl.addWidget(v, 0, Qt.AlignRight)
            p4.layout().addWidget(row)
            self.live[key] = v

        # ⚠️ 判机位必须看「观察窗口内的摆幅」，不能看当前值：
        #    家长多半在孩子正常坐着时打开这一页，那时差值天然是 0，
        #    拿当前值判会误报"机位不对"。
        #    用摆幅（max - min）还有个好处：**不需要基线**，所以不用等标定完成就能验机位。
        self._max_neck = None
        self._min_neck = None
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 8, 0, 3)
        n = QLabel("颈部前倾角摆幅（本次观察）")
        n.setStyleSheet("font-weight:700;")
        rl.addWidget(n, 1)
        self.live_max = QLabel("—")
        self.live_max.setObjectName("metricValue")
        rl.addWidget(self.live_max, 0, Qt.AlignRight)
        p4.layout().addWidget(row)

        self.live_verdict = QLabel("")
        self.live_verdict.setWordWrap(True)
        p4.layout().addWidget(self.live_verdict)

        btn_reset = QPushButton("开始观察 / 重置最大值")
        btn_reset.setObjectName("ghost")
        btn_reset.setCursor(Qt.PointingHandCursor)
        btn_reset.clicked.connect(self._reset_live)
        p4.layout().addWidget(btn_reset, 0, Qt.AlignLeft)
        p4.layout().addStretch(1)

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(300)
        self._live_timer.timeout.connect(self._refresh_live)
        self._live_timer.start()
        self._refresh_live()

        # --- 页5 审核判定（标定阈值的唯一现实途径）---
        p5 = self._tab_scroll(tabs, "审核判定")
        self.review_panel = ReviewPanel(parent, p5)
        p5.layout().addWidget(self.review_panel)

        # --- 页6 标定自检 ---
        p6 = self._tab_scroll(tabs, "标定自检")
        self.sanity = QTextEdit()
        self.sanity.setReadOnly(True)
        p5.layout().addWidget(self.sanity)
        b = QPushButton("刷新自检结果")
        b.setObjectName("ghost")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self._refresh_sanity)
        p5.layout().addWidget(b, 0, Qt.AlignLeft)
        self._refresh_sanity()

        # --- 页7 日志 ---
        p7 = self._tab_scroll(tabs, "运行日志")
        self.runlog_tip = QLabel()
        self.runlog_tip.setObjectName("hint")
        self.runlog_tip.setWordWrap(True)
        self.runlog_tip.setText(
            f"日志文件：{parent.runlog.path}" if parent.runlog.path
            else "本次未写日志文件（--no-log，或日志目录不可写）。")
        p7.layout().addWidget(self.runlog_tip)
        self.logwin = QTextEdit()
        self.logwin.setObjectName("logwin")
        self.logwin.setReadOnly(True)
        self.logwin.setFont(QFont("Consolas", 9))
        self.logwin.setPlainText("\n".join(parent._log_lines))
        p7.layout().addWidget(self.logwin)

        # --- 底部 ---
        foot = QHBoxLayout()
        btn_save = QPushButton("保存并应用")
        btn_save.setObjectName("save")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.clicked.connect(self._on_save)
        foot.addWidget(btn_save)

        btn_reset = QPushButton("恢复默认")
        btn_reset.setObjectName("ghost")
        btn_reset.setCursor(Qt.PointingHandCursor)
        btn_reset.clicked.connect(self._on_reset)
        foot.addWidget(btn_reset)
        foot.addStretch(1)

        btn_close = QPushButton("关闭")
        btn_close.setObjectName("ghost")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(self.accept)
        foot.addWidget(btn_close)
        root.addLayout(foot)

    def _tab_scroll(self, tabs, title):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(page)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        tabs.addTab(scroll, f"  {title}  ")
        return page

    def _add_row(self, form, key, label, value, kind, hint, box=None):
        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        edit = QLineEdit("" if value is None else str(value))
        edit.setFixedWidth(140)
        v.addWidget(edit)
        h = QLabel(hint)
        h.setObjectName("hint")
        h.setWordWrap(True)
        v.addWidget(h)
        form.addRow(label, holder)
        (box if box is not None else self._inputs)[key] = (edit, kind)

    def _reset_live(self):
        self._max_neck = None
        self._min_neck = None
        self._refresh_live()

    # ---------------- 试听

    @staticmethod
    def _available_voices():
        """扫描可用的语音片段名（mp3 与 wav 合并，排除音量缓存/归档子目录）。"""
        return pe.Speaker.available()

    def _preview_voice(self, name):
        """试听一条提醒语。忽略静音开关 —— 点了没声音会让人以为功能坏了。

        ⚠️ 播放是**异步**的（say() 只入队），所以"有没有文件"要自己查，
        不能拿 say() 的返回值当"播放成功"。
        """
        app = self.parent()
        vol = next((lv for lv, r in self.radios.items() if r.isChecked()), "中")
        eng = getattr(app, "engine", None)
        if eng is not None:
            eng.preview_voice(name, vol)
        else:
            sp = getattr(app, "_demo_speaker", None) or pe.Speaker(volume=vol)
            sp.mute = False
            if vol in sp.GAIN:
                sp.volume = vol
            sp.say(name)
        have = any(os.path.isfile(os.path.join(pm.VOICE_DIR, name + e))
                   for e in (".mp3", ".wav"))
        if hasattr(self, "lbl_preview"):
            self.lbl_preview.setText(
                f"已排入播放队列：「{name}」（音量 {vol}）"
                if have else f"没找到 {name}.mp3/.wav，会退回系统提示音")
        if hasattr(app, "_log"):
            app._log(f"[家长设置] 试听语音「{name}」音量={vol}")

    def _set_verdict(self, msg, color="#7A8798"):
        self.live_verdict.setText(msg)
        self.live_verdict.setStyleSheet(f"font-weight:700; color:{color};")

    def _clear_values(self):
        for v in self.live.values():
            v.setText("—")
        self.live_max.setText("—")

    def _refresh_live(self):
        app = self.parent()
        eng = getattr(app, "engine", None)
        snap = eng.snapshot() if eng else None
        try:
            base = pe.load_baseline(pe.load_config()) or {}
        except Exception:
            base = {}

        def dead(msg):
            self._clear_values()
            self._set_verdict(msg)

        if snap is None:
            dead("演示模式／引擎未运行，没有实时数据。")
            return
        if snap.get("phase") not in ("monitoring", "calibrating"):
            dead(f"引擎当前状态：{snap.get('phase')}（还没进入监测）。")
            return

        m = snap.get("metrics")
        if not m:
            dead("当前画面里没检测到人。让孩子的头和肩膀进画面。")
            return

        for key, lbl in self.live.items():
            cur = m.get(key)
            b = base.get(key)
            if cur is None:
                lbl.setText("拿不到")
            elif b is None:
                lbl.setText(f"{cur:.1f}")
            else:
                lbl.setText(f"{cur:.1f}   （基线 {b:.1f}，差 {cur - b:+.1f}）")

        cur_n = m.get("neck_angle")
        if cur_n is None:
            self.live_max.setText("—")
            self._set_verdict("")
            return
        self._max_neck = cur_n if self._max_neck is None else max(self._max_neck, cur_n)
        self._min_neck = cur_n if self._min_neck is None else min(self._min_neck, cur_n)
        swing = self._max_neck - self._min_neck
        extra = ""
        b_n = base.get("neck_angle")
        if b_n is not None:
            extra = f"；相对基线最大 {self._max_neck - b_n:+.1f}°"
        self.live_max.setText(
            f"{swing:.1f}°   （{self._min_neck:.1f}° ~ {self._max_neck:.1f}°{extra}）")

        # ⚠️ 判据用「摆幅」而不是「相对基线」—— 这样不需要先标定就能验机位。
        if swing >= 10:
            self._set_verdict(f"✅ 机位没问题。孩子前伸让角度变化了 {swing:.1f}°，"
                              "前后关系确实测得到。", "#1E8A4C")
        elif swing >= 5:
            self._set_verdict(f"⚠️ 勉强可用（摆幅 {swing:.1f}°）。建议再往侧面转一点，"
                              "灵敏度会明显更好。", "#B4701A")
        else:
            self._set_verdict(f"❌ 摆幅只有 {swing:.1f}° —— 摄像头太正对孩子了。"
                              "请把摄像头转向孩子侧面，机位摆到 80~90° 正侧。"
                              "（若还没让孩子前伸过，先让他前伸一下再看。）", "#B4701A")

    def _refresh_sanity(self):
        try:
            base = pe.load_baseline(pe.load_config())
        except Exception:
            base = None
        if not base:
            self.sanity.setPlainText("还没有基线。第一次启动会自动学习，"
                                     "或在主界面点「重新学习坐姿」。\n")
            return
        lines = []
        for label, v, verdict, hint in pm.sanity_rows(["侧面"], [base]):
            mark = {"ok": "✅", "warn": "⚠️ ", "info": "ℹ️ "}[verdict]
            val = "（拿不到）" if v is None else f"{v:.3f}"
            lines.append(f"{mark} {label} = {val}")
            if verdict in ("warn", "info"):
                lines.append(f"      {hint}")
        self.sanity.setPlainText("\n".join(lines))

    def _on_save(self):
        try:
            new = pe.load_config()
            for key, (edit, kind) in self._inputs.items():
                if key in ("side_thresholds",):
                    continue
                new[key] = kind(edit.text().strip())
            for key, (edit, _) in self._inputs.get("side_thresholds", {}).items():
                new["side_thresholds"][key] = float(edit.text().strip())
            new["mute"] = self.chk_mute.isChecked()
            new["volume"] = next((lv for lv, r in self.radios.items() if r.isChecked()), "中")
        except ValueError:
            QMessageBox.critical(self, "填错了", "请检查：数字项里不能有文字或空格。")
            return

        if not pe.save_config(new):
            QMessageBox.critical(self, "保存失败", "写不进 config.json，请检查目录权限。")
            return

        app = self.parent()
        if getattr(app, "engine", None):
            app.engine.cfg.update(new)
            pe.apply_config(app.engine.cfg)
            app.engine.set_audio(mute=new["mute"], volume=new["volume"])
        QMessageBox.information(
            self, "已保存",
            "参数已保存。\n\n"
            "提醒节奏、声音、阈值即时生效；\n"
            "摄像头索引、偏航角、帧率需要重启程序（或点「重新学习坐姿」）才完全生效。")

    def _on_reset(self):
        ok = QMessageBox.question(self, "恢复默认", "把所有参数恢复成默认值？重启程序后生效。")
        if ok == QMessageBox.Yes:
            pe.save_config(pe.DEFAULT_CONFIG)


# ---------------------------------------------------------------- 入口

def _fake_snapshot(phase, **kw):
    s = {"phase": phase, "error": None, "preview": None, "kps": None,
         "level": "idle", "text": pe.TEXT_IDLE, "issue": None, "pending": None,
         "calib": {"done": 0, "need": 15, "sec_left": 0.0},
         "baseline": None, "detect_rate": None, "metrics": None,
         "stats": pe.PostureEngine._blank_stats()}
    s.update(kw)
    return s


class _StubEngine:
    """给自检用的假引擎：回一个固定快照，并记录 pause/resume 调用。"""

    def __init__(self, snap):
        self._s = snap
        self.calls = []

    def snapshot(self):
        return self._s

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")


def selftest(app):
    """把每个 phase 都渲染一遍，再用 grab() 强制走一次 paintEvent。"""
    app.update()
    app.demo = True        # 走「非引擎」分支，避免再读 self.engine
    cases = [
        _fake_snapshot("idle"),
        _fake_snapshot("opening"),
        _fake_snapshot("calibrating", calib={"done": 6, "need": 15, "sec_left": 4.0}),
        _fake_snapshot("monitoring", level="good", text=pe.TEXT_GOOD,
                       baseline={"neck_angle": 3.0, "trunk_angle": None}, detect_rate=0.92,
                       stats={"total_sec": 600, "good_sec": 520, "good_ratio": 0.87,
                              "reminders": 3, "streak_sec": 130, "best_streak_sec": 490}),
        _fake_snapshot("monitoring", level="warn", text="头抬高一点点～",
                       pending=("头前倾", 2.4), detect_rate=0.88),
        _fake_snapshot("monitoring", level="nodetect", text=pe.TEXT_NODETECT,
                       detect_rate=0.12),
        _fake_snapshot("error", error="标定失败：只采到 3 个有效样本。"),
    ]
    checks = 0
    for s in cases:
        app._render_snapshot(s)
        app.update()
        pix = app.grab()                      # 强制走 paintEvent，能顶出绘制异常
        assert not pix.isNull(), "界面抓图为空"
        checks += 1

    # 预览渲染路径（用合成画面）
    frame = app._demo_frame()
    if frame is not None:
        app._render_preview(frame)
        app.update()
        checks += 1

    # 家长设置窗口
    dlg = ParentDialog(app)
    dlg.show()
    app.update()
    assert not dlg.grab().isNull(), "家长设置窗口抓图为空"
    checks += 1

    # 「机位检查」的三档判定 —— 这是判断"侧面机位摆对没"的核心逻辑，必须锁住。
    # 判据是**摆幅**（观察窗口内 max-min），所以每个用例要走两步：先静止，再前伸。
    orig_lb = pe.load_baseline
    pe.load_baseline = lambda cfg=None: {"neck_angle": 4.0, "trunk_angle": None,
                                         "sh_ear_len": 210.0}
    try:
        for swing, expect in ((3.0, "太正对孩子"), (7.0, "勉强可用"), (13.0, "机位没问题")):
            def _snap(neck):
                return _fake_snapshot("monitoring", level="good", text=pe.TEXT_GOOD,
                                      metrics={"neck_angle": neck, "sh_ear_len": 200.0,
                                               "trunk_angle": None})
            dlg._reset_live()
            app.engine = _StubEngine(_snap(5.0))          # 静止
            dlg._refresh_live()
            app.engine = _StubEngine(_snap(5.0 + swing))  # 前伸
            dlg._refresh_live()
            got = dlg.live_verdict.text()
            assert expect in got, f"摆幅={swing} 期望判定含「{expect}」，实际：{got}"
            checks += 1
    finally:
        pe.load_baseline = orig_lb
        app.engine = None

    # 审核面板能渲染（空数据也要不炸）
    dlg.review_panel.reload()
    app.update()
    checks += 1

    # 审核存图闭环：存 -> 列 -> 标注 -> 统计 -> 阈值建议 -> 清空
    # 用临时目录，别在项目里留测试文件
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="posture_review_")
    try:
        st = pe.ReviewStore(root=tmp, max_files=50, enabled=True)
        base = {"neck_angle": 5.0, "sh_ear_len": 210.0, "trunk_angle": None}
        frame = app._demo_frame()
        for neck in (20.0, 22.0, 6.0, 7.0):
            st.save(frame, {"neck_angle": neck, "sh_ear_len": 200.0}, base,
                    "头前倾", kind="reminder", cfg={"camera": 0, "side_yaw": 90})
        assert len(st.items("pending")) == 4, "应有 4 条待审"
        for it in st.items("pending"):
            # 20/22 分 -> 该报警；6/7 分 -> 误报
            st.label(it, it["data"]["metrics"]["neck_angle"] > 10)
        s = st.stats()
        assert (s["pending"], s["labeled"], s["wrong"]) == (0, 4, 2), s

        r = st.suggest_all(pe.load_config()).get("neck_angle")
        assert r is not None, "应能给出 neck_angle 的建议阈值"
        assert (r["fp_at_suggested"], r["fn_at_suggested"]) == (0, 0), r
        assert 2.0 <= r["suggested"] <= 15.0, r
        checks += 1

        assert st.clear_all() > 0, "清空应删到文件"
        assert st.items("pending") == [] and st.items("done") == [], "清空后应无残留"
        checks += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 坐正了不发声：跑满两轮演示序列，验证**只有 warn** 才发声音
    calls = []

    class _StubSpeaker:
        mute = False

        def say(self, issue):
            calls.append(("say", issue))
            return True

        def chime(self, kind="good"):
            calls.append(("chime", kind))

    orig_sp = app._demo_speaker
    app._demo_speaker = _StubSpeaker()
    app._demo_i = 0
    for _ in range(10):          # 一轮 5 步 = good,good,warn,warn,nodetect
        app.demo_step()
    assert all(c[0] == "say" for c in calls), f"坐正了不应发出任何声音，实际 {calls}"
    assert len(calls) == 4, f"2 轮 × 2 个 warn 应发声 4 次，实际 {len(calls)}（{calls}）"
    app._demo_speaker = orig_sp
    checks += 1

    # 空格键 = 暂停 / 继续（需求功能，必须锁住）
    stub = _StubEngine(_fake_snapshot("monitoring", level="good", text=pe.TEXT_GOOD))
    was_demo, was_engine = app.demo, app.engine
    app.demo, app.engine = False, stub
    app.btn_toggle.setText("暂停")

    def press_space(autorep=False):
        QApplication.sendEvent(app, QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key_Space, Qt.NoModifier, "", autorep))

    press_space()
    assert stub.calls == ["pause"], f"空格应暂停引擎，实际 {stub.calls}"
    assert app.btn_toggle.text() == "继续", app.btn_toggle.text()
    press_space()
    assert stub.calls == ["pause", "resume"], f"空格应继续引擎，实际 {stub.calls}"
    assert app.btn_toggle.text() == "暂停", app.btn_toggle.text()
    press_space(autorep=True)
    assert stub.calls == ["pause", "resume"], f"长按不应重复触发，实际 {stub.calls}"
    app.demo, app.engine = was_demo, was_engine
    app.btn_toggle.setText("暂停")
    checks += 1

    dlg.close()

    print(f"界面自检通过：{checks} 项（7 个状态分支 + 预览渲染 + 家长设置窗口 "
          f"+ 机位判定 3 档 + 审核面板 + 存图标注闭环 + 清空 + 坐正不发声 "
          f"+ 空格键暂停/继续 + 语音片段 {len(ParentDialog._available_voices())} 条）")
    app.close()


def main():
    ap = argparse.ArgumentParser(description="坐姿小卫士 · 图形界面（PySide6）")
    ap.add_argument("--demo", action="store_true", help="演示模式：不连摄像头，循环展示状态")
    ap.add_argument("--selftest", action="store_true", help="只构建界面并退出（离屏冒烟测试）")
    ap.add_argument("--screen", action="store_true",
                    help="打印屏幕参数（逻辑尺寸/可用区域/缩放）后退出，不显示窗口")
    ap.add_argument("--log-file", default=None,
                    help="运行日志文件路径（默认 logs\\run-YYYYMMDD.log，按天一个，保留 7 天）")
    ap.add_argument("--no-log", action="store_true", help="不写运行日志文件")
    args = ap.parse_args()

    if args.screen:
        # 真机屏幕参数只能用真实平台查。
        # ⚠️ 别用 --selftest 看屏幕参数 —— 它强制走 offscreen 平台，报的是 800x800 假数据。
        app = QApplication(sys.argv)
        for s in QApplication.screens():
            g, a = s.geometry(), s.availableGeometry()
            print(f"屏幕 {s.name()}: 逻辑几何 {g.width()}x{g.height()}   "
                  f"可用区域 {a.width()}x{a.height()}   "
                  f"缩放 {s.devicePixelRatio():.2f}x   DPI {s.logicalDotsPerInch():.0f}")
        return

    if args.selftest:
        # 离屏跑，别在用户屏幕上闪窗
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication(sys.argv)
    chosen = apply_app_font(app)
    app.setStyleSheet(QSS)
    if args.selftest:
        # ⚠️ 离屏平台（offscreen）枚举到 0 个字体族 —— Qt 在该模式下找不到自己的字体目录。
        #    所以这里必然是"系统默认"，不代表真实运行时选不中雅黑。别把它当故障查。
        print(f"字体：{chosen or '（离屏平台枚举不到字体族；真实运行时用系统字体）'}")
        scr = QApplication.primaryScreen()
        if scr is not None:
            av = scr.availableGeometry()
            print(f"屏幕：可用区域 {av.width()}x{av.height()}（逻辑像素），"
                  f"缩放 {scr.devicePixelRatio():.2f}")

    win = PostureApp(demo=args.demo, autostart=not args.selftest,
                     log_path=args.log_file, log_enabled=not args.no_log)
    win.show()
    if args.selftest:
        print(f"窗口：{win.width()}x{win.height()}  "
              f"（按屏幕可用区域收敛，HDPI 下不会超出屏幕）")
        selftest(win)
        return
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

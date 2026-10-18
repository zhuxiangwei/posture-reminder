# -*- coding: utf-8 -*-
"""主窗口：状态卡 + 摄像头预览 + 今日表现 + 底部按钮。

代码原样搬自 `posture_app.py`。
"""

import time

from PySide6.QtCore import (Qt, QTimer)
from PySide6.QtGui import (QFont, QImage, QPixmap)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QMessageBox, QProgressBar, QPushButton, QSizePolicy,
                               QTextEdit, QVBoxLayout, QWidget)
from posture import deps, engine, gate, logutil, pose, settings, texts, voice
from posture.ui.parent_dialog import ParentDialog
from posture.ui.theme import LEVEL_STYLE, fmt_dur
from posture.ui.widgets import FaceWidget, ScaledImageLabel


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
        self.runlog = logutil.RunLog(log_path, enabled=log_enabled)
        self._demo_stats = engine.PostureEngine._blank_stats()
        self._demo_i = 0
        # 开流计时（界面侧）：从"看到 opening"那一刻起算，只用于「已经等了 N 秒」这句反馈。
        # ⚠️ 不读引擎的时间：引擎是 300ms 后才 start 的，两个起点不一样，
        #    而且真开流耗时该由日志/家长设置报，这里只要给孩子一个"没卡死"的信号。
        self._cam_t0 = None

        self.setWindowTitle("坐姿小卫士" + ("（演示）" if demo else ""))
        w, h = self._fit_to_screen(900, 620)
        self.resize(w, h)
        self.setMinimumSize(700, 500)

        self._build_ui()

        # ⚠️ 声音按钮的初始文字必须读配置，不能写死"声音：开"。
        #    否则存过静音之后，按钮显示的状态是错的（显示"开"其实静音着）。
        cfg0 = settings.load_config()
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
        self.lbl_big = QLabel(texts.TEXT_IDLE)
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

        # ---- 摄像头状态横幅（降档 / 掉线**必须**显式可见）----
        # 为什么单独一条，而不是塞进状态卡的大字下面：
        #   ① 状态卡是给**孩子**看的，内容是"坐得真棒 / 头抬高一点"，
        #      设备问题孩子既看不懂也解决不了，混在一起只会稀释提醒；
        #   ② 这条是给**家长**看的，而且必须**常驻** ——
        #      用户口径是「坚持 4K 优先、不许静默降档」，
        #      实际只拿到 1080p 时如果不说，家长永远以为清晰度没问题。
        # 所以：只在有内容时显示，且放在状态卡与预览之间（一进窗口就在视线路径上）。
        self.lbl_cam = QLabel("")
        self.lbl_cam.setObjectName("camWarn")
        self.lbl_cam.setWordWrap(True)
        self.lbl_cam.hide()
        root.addWidget(self.lbl_cam)

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
        self.preview = ScaledImageLabel("等待画面…")
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

        self._apply_level("idle", texts.TEXT_IDLE)

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

    # ---------------- 摄像头状态（开流进度 / 降档 / 掉线重连）

    CAM_TONE = {
        # 降档：暖橙（提示但不批评）
        "warn": ("#FDF3E1", "#8A5A12", "#F0DCB4"),
        # 断线：灰蓝（设备的问题，不是孩子的问题）
        "bad": ("#EDF1F7", "#4C5A6B", "#D5DDE8"),
    }

    def _cam_trying(self, cam):
        """开流期间"正在试哪个组合"（后端 ／ 档位 ／ 第几轮）。

        数据源是 `cam["trying"]` —— 采集线程每起一轮开流就更新一次。
        老快照（或假数据）没这个键时退回后端+档位，再退回空串；
        退回逻辑必须在这边做，否则"正在试"这句话会在没数据时显示成 `NonexNone`。
        """
        t = cam.get("trying") or {}
        parts = [t.get("backend") or cam.get("backend")]
        w = t.get("width") or cam.get("width")
        h = t.get("height") or cam.get("height")
        if w and h:
            parts.append(f"{w}x{h}")
        i = t.get("camera")
        if i is not None:
            parts.append(f"摄像头 {i}")
        n = int(t.get("attempt") or 0)
        if n > 1:
            parts.append(f"第 {n} 轮")
        return " ／ ".join(str(p) for p in parts if p)

    def _cam_break_note(self, cam):
        """把"为什么没画面、重连到第几次了"收成一句人话。

        ⚠️ 这里刻意**不**写"看不到你啦"：那句话是"人在画面外"的意思，
           设备掉线说成这个，家长会去挪摄像头，而真正该查的是 USB / 占用。
        """
        bits = []
        stale = cam.get("stale_sec")
        if isinstance(stale, (int, float)) and stale >= 1:
            bits.append(f"已经 {stale:.0f} 秒没有新画面")
        n = int(cam.get("reconnects") or 0)
        if cam.get("phase") == "reconnecting":
            bits.append(f"正在自动重连（第 {n + 1} 次尝试）")
        elif n:
            bits.append(f"已自动重连 {n} 次")
        if cam.get("short"):
            bits.append(str(cam["short"]))
        return "；".join(bits)

    def _render_cam_banner(self, s):
        """有降档/掉线才显示；没有就整条隐藏（平时不占位、不吓人）。"""
        cam = s.get("cam") or {}
        phase = s.get("phase", "idle")
        lines, tone = [], "warn"
        # idle/stopped/opening/error 不显示横幅：
        #   opening 的结果还没出来（进度写在状态卡里）；
        #   error 分支自己会把 short + 排查清单写进状态卡与提示行，不重复。
        if phase in ("calibrating", "monitoring"):
            cp = cam.get("phase")
            if cp in ("reconnecting", "failed"):
                tone = "bad"
                lines.append(texts.TEXT_RECONNECTING if cp == "reconnecting"
                             else texts.TEXT_CAM_FAILED)
                note = self._cam_break_note(cam)
                if note:
                    lines.append(note)
            else:
                # ★ 不许静默降档：引擎把 OpenResult.warnings 原样递上来
                lines = [str(w) for w in (s.get("cam_warnings") or []) if w]
                if not lines and cam.get("degraded"):
                    lines.append(f"实际画面只有 {cam.get('width')}x{cam.get('height')}，"
                                 "低于请求档位（判定仍可用，但现场照片会不清楚）。")
        if not lines:
            self.lbl_cam.hide()
            return
        bg, fg, bd = self.CAM_TONE[tone]
        self.lbl_cam.setStyleSheet(
            f"#camWarn {{ background: {bg}; color: {fg};"
            f" border: 1px solid {bd}; border-radius: 10px; padding: 8px 12px;"
            f" font-size: 12px; }}")
        self.lbl_cam.setText("⚠️ " + "\n".join(lines[:3]))
        self.lbl_cam.show()

    def _render_preview(self, frame, kps=None):
        if frame is None or deps.cv2 is None:
            return
        try:
            img = frame
            if kps is not None:
                img = frame.copy()
                h, w = img.shape[:2]
                for i in pose.KEYPOINTS_DRAW:
                    deps.cv2.circle(img, (int(kps[i].x * w), int(kps[i].y * h)), 3,
                                  (0, 220, 0), -1)
            if not img.flags["C_CONTIGUOUS"]:
                img = img.copy()
            h, w, ch = img.shape
            # QImage 不持有 numpy 缓冲，必须 copy() 后再交给 QPixmap
            qimg = QImage(img.data, w, h, ch * w, QImage.Format_BGR888).copy()
            # 只交出源图，缩放由控件在**绘制时**按当前尺寸做。
            # ⚠️ 不能在这里先缩放到控件尺寸：预览区高度会随表头提示语的行数变化
            #    （一行 vs 两行 → 布局回流），提前缩放会把尺寸固化下来，
            #    控件变大后图片只占一角（实测就是"只显示四分之一"，且时好时坏）。
            self.preview.set_image(QPixmap.fromImage(qimg))
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
        self.engine = engine.PostureEngine(cfg=settings.load_config(), on_log=self._log)
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
        cfg = settings.load_config()
        cfg["mute"] = turn_off
        settings.save_config(cfg)
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
            # ⚠️ 必须等引擎线程真正收尾再往下走。
            #    否则解释器开始拆除时，引擎可能正卡在 detect_for_video 里，
            #    mediapipe 的 ThreadPoolExecutor 已被 atexit 关掉，会抛
            #        RuntimeError: cannot schedule new futures after shutdown
            #    不影响功能（异常被引擎的兜底捕获），但会往日志里写一段
            #    看不懂的栈，并且拖慢进程退出（实测曾残留上百秒）。
            #    join 带超时，卡住也不会把界面挂死（引擎是 daemon 线程）。
            try:
                self.engine.join(timeout=3.0)
            except Exception:
                pass
        self._timer.stop()
        super().closeEvent(ev)

    # ---------------- 轮询

    def _tick(self):
        if self.engine:
            self._render_snapshot(self.engine.snapshot())

    def _render_snapshot(self, s):
        phase = s.get("phase", "idle")
        self._render_preview(s.get("preview"), s.get("kps"))

        # 开流计时与摄像头横幅：**每个分支都要生效**（error / 暂停分支会提前 return，
        # 所以放在最前面统一处理，而不是在每个分支里重复一遍）
        if phase == "opening":
            if self._cam_t0 is None:
                self._cam_t0 = time.perf_counter()
        else:
            self._cam_t0 = None
        self._render_cam_banner(s)

        if phase == "error":
            self._apply_level("warn", "有点小问题", s.get("error") or "")
            self.btn_toggle.setText("重试")
            # 有详情就显示详情（那是 camera.CHECKLIST 那种"照着做"的排查顺序），
            # 没有才退回这段通用话术。重构前失败时只有一坨清单、没有短标题，
            # 界面弹窗根本放不下 —— 现在反过来：一行标题 + 可展开的清单。
            self.lbl_hint.setText(
                s.get("error_detail")
                or "修好后点「重试」。最常见的两个原因：摄像头被别的程序占用（会议软件／"
                   "相机 App），或画面里始终看不到人导致标定失败。")
            return

        if phase == "opening":
            cam = s.get("cam") or {}
            waited = time.perf_counter() - (self._cam_t0 or time.perf_counter())
            trying = self._cam_trying(cam)
            sub = f"已经等了 {waited:.0f} 秒" + (f"，正在试 {trying}" if trying else "")
            self._apply_level("idle", s.get("text") or texts.TEXT_OPENING, sub)
            self.lbl_hint.setText(
                "窗口没有卡住，正在探测摄像机的能力（第一次最慢，之后会记住结果、快很多）。"
                "每一步都写在「家长设置 → 运行日志」里。")
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
            self._apply_level("idle", texts.TEXT_IDLE)
            return

        if self.btn_toggle.text() == "继续":
            self._apply_level("idle", texts.TEXT_PAUSED, "点「继续」接着看")
            return

        level = s.get("level", "idle")
        sub = ""
        pend = s.get("pending")
        if pend:
            need = float(self.engine.cfg.get("sustain_sec", gate.SUSTAIN_SEC)) if self.engine \
                else gate.SUSTAIN_SEC
            sub = f"再保持 {max(0.0, need - pend[1]):.0f} 秒，会自动提醒一下"
        if level == "nodetect":
            sub = "请坐到座位中间，让摄像头看得到你"
        # 设备侧优先于"坐姿"叙事：没画面时不该说"请坐到座位中间"（孩子照做也没用），
        # 更不该让孩子以为是自己坐错了 —— 引擎这时给的 text 已经是「正在自动重连」。
        cam = s.get("cam") or {}
        if cam.get("phase") in ("reconnecting", "failed"):
            sub = self._cam_break_note(cam)
        self._apply_level(level, s.get("text") or texts.TEXT_IDLE, sub)

        st = s.get("stats") or engine.PostureEngine._blank_stats()
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
        cfg = settings.load_config()
        self._demo_speaker = voice.Speaker(mute=bool(cfg.get("mute")),
                                        volume=cfg.get("volume", "中"))
        mute = "静音中" if cfg.get("mute") else f"开（音量档位：{cfg.get('volume', '中')}）"
        self._log(f"演示模式：不连摄像头，循环展示各种状态；声音 {mute}")
        self.lbl_hint.setText(
            "演示模式 —— 正在循环展示「坐得好／提醒／看不到人」三种状态，"
            f"语音提醒也会播。当前声音：{mute}（点下面「声音」按钮可切换）。"
            "去掉 --demo 参数即可正常使用。")

    def demo_step(self):
        # (状态, 大字, 副标题, 对应的问题名) —— 问题名用来取 voice/<名字>.wav
        seq = [("good", texts.TEXT_GOOD, "", None),
               ("good", texts.TEXT_GOOD, "", None),
               ("warn", "头抬高一点点～", "再保持 2 秒，会自动提醒一下", "头前倾"),
               ("warn", "背挺直一点～", "再保持 1 秒，会自动提醒一下", "塌腰"),
               ("nodetect", texts.TEXT_NODETECT, "请坐到座位中间，让摄像头看得到你", None)]
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
            deps.cv2.rectangle(f, (150, 62), (330, 250), (206, 214, 224), -1)
            deps.cv2.circle(f, (240, 110), 42, (188, 198, 210), -1)
            deps.cv2.putText(f, "DEMO", (16, 250), deps.cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (150, 160, 175), 2)
            return f
        except Exception:
            return None

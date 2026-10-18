# -*- coding: utf-8 -*-
"""家长设置对话框（阈值 / 机位 / 提醒与声音 / 机位检查 / 审核 / 日志）。

代码原样搬自 `posture_app.py`。
"""

import os

from PySide6.QtCore import (Qt, QTimer)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFormLayout, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QRadioButton, QScrollArea, QTabWidget, QTextEdit,
                               QVBoxLayout, QWidget)
from posture import metrics as mx, paths, settings, voice
from posture.ui.review_panel import ReviewPanel


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
        cfg = settings.load_config()
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

        # ---- 设备实况（家长要看的是**实际拿到了什么**，不是请求了什么）----
        # 为什么放这一页：机位检查是家长唯一会主动"盯着数值看"的页面，
        # 而"实际只有 1080p / 已经自动重连 3 次 / 没协商到 MJPG"这几条
        # 恰好是解释"为什么照片糊、为什么刚才断了一下"的线索。
        self.cam_state = QLabel()
        self.cam_state.setObjectName("hint")
        self.cam_state.setWordWrap(True)
        p4.layout().addWidget(self.cam_state)

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
        # ⚠️ 这两件控件必须加到 **p6**：早期写的 p5 是复制粘贴留下的错，
        #    结果是「审核判定」页底下莫名多出一块自检文本，而「标定自检」页是空的。
        p6 = self._tab_scroll(tabs, "标定自检")
        self.sanity = QTextEdit()
        self.sanity.setReadOnly(True)
        p6.layout().addWidget(self.sanity)
        b = QPushButton("刷新自检结果")
        b.setObjectName("ghost")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self._refresh_sanity)
        p6.layout().addWidget(b, 0, Qt.AlignLeft)
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
        return voice.Speaker.available()

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
            sp = getattr(app, "_demo_speaker", None) or voice.Speaker(volume=vol)
            sp.mute = False
            if vol in sp.GAIN:
                sp.volume = vol
            sp.say(name)
        have = any(os.path.isfile(os.path.join(paths.VOICE_DIR, name + e))
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

    def _cam_state_text(self, snap):
        """家长设置里的设备实况：实际档位、后端、降档原因、重连次数。

        为什么单独一个方法：这一行要**无条件**刷新（引擎进不了监测时更要看），
        而 `self.live` 那几项在无画面时会被 `dead()` 清掉 —— 两者生命周期不同，
        混在一起写就会出现"断线时反而什么都看不见"。
        """
        if snap is None:
            return "摄像头：引擎未运行（演示模式），没有设备信息。"
        cam = snap.get("cam") or {}
        if not cam.get("phase"):
            return f"摄像头：还没开流（当前阶段 {snap.get('phase')}）。"
        # ⚠️ 开流成功之前 `backend / width / height` 全是 None —— 档位是"开出来那一刻"
        #    才知道的。这时不许拼成「实际 NonexNone」：家长页出现 None 会让人以为程序坏了。
        if cam.get("width"):
            bits = [f"后端 {cam.get('backend')} · 索引 {cam.get('camera')} · "
                    f"实际 {cam['width']}x{cam.get('height')}",
                    f"状态 {cam.get('phase')}"]
        else:
            bits = [f"状态 {cam.get('phase')}（还没开出画面，档位要开成功才知道）"]
        # 开流/重连进行中：把"此刻在试哪个组合"也写出来（排查"加载很久"最有用的一行）。
        # 格式化逻辑复用主窗口的 `_cam_trying`，别让同一句话在两处各写一遍、各自漂移。
        if cam.get("phase") in ("opening", "reconnecting"):
            f = getattr(self.parent(), "_cam_trying", None)
            t = f(cam) if f else ""
            if t:
                bits.append(f"正在试 {t}")
        if cam.get("degraded"):
            bits.append("⚠️ 低于请求档位")
        n = int(cam.get("reconnects") or 0)
        if n:
            bits.append(f"已自动重连 {n} 次")
        txt = " ｜ ".join(bits)
        for w in (snap.get("cam_warnings") or []):
            txt += f"\n· {w}"
        if cam.get("short"):
            txt += f"\n· {cam['short']}"
        return txt

    def _refresh_live(self):
        app = self.parent()
        eng = getattr(app, "engine", None)
        snap = eng.snapshot() if eng else None
        self.cam_state.setText(self._cam_state_text(snap))
        try:
            # ⚠️ 用引擎的**实际**采集条件（`_eff`）来读基线，不能用 config 的请求值：
            #    基线是按"实际拿到的分辨率"标定并校验的，请求 4K 只拿到 1080p 时，
            #    拿请求值去读会把一份好基线判成无效（家长看到的是"还没标定"）。
            cfg = getattr(eng, "_eff", None) or settings.load_config()
            base = settings.load_baseline(cfg) or {}
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
            base = settings.load_baseline(settings.load_config())
        except Exception:
            base = None
        if not base:
            self.sanity.setPlainText("还没有基线。第一次启动会自动学习，"
                                     "或在主界面点「重新学习坐姿」。\n")
            return
        lines = []
        for label, v, verdict, hint in mx.sanity_rows(["侧面"], [base]):
            mark = {"ok": "✅", "warn": "⚠️ ", "info": "ℹ️ "}[verdict]
            val = "（拿不到）" if v is None else f"{v:.3f}"
            lines.append(f"{mark} {label} = {val}")
            if verdict in ("warn", "info"):
                lines.append(f"      {hint}")
        self.sanity.setPlainText("\n".join(lines))

    def _on_save(self):
        try:
            new = settings.load_config()
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

        if not settings.save_config(new):
            QMessageBox.critical(self, "保存失败", "写不进 config.json，请检查目录权限。")
            return

        app = self.parent()
        if getattr(app, "engine", None):
            app.engine.cfg.update(new)
            settings.apply_config(app.engine.cfg)
            app.engine.set_audio(mute=new["mute"], volume=new["volume"])
        QMessageBox.information(
            self, "已保存",
            "参数已保存。\n\n"
            "提醒节奏、声音、阈值即时生效；\n"
            "摄像头索引、偏航角、帧率需要重启程序（或点「重新学习坐姿」）才完全生效。")

    def _on_reset(self):
        ok = QMessageBox.question(self, "恢复默认", "把所有参数恢复成默认值？重启程序后生效。")
        if ok == QMessageBox.Yes:
            settings.save_config(settings.DEFAULT_CONFIG)

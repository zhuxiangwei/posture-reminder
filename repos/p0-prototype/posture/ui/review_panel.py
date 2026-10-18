# -*- coding: utf-8 -*-
"""家长「审核判定」面板：审核程序判过的现场，用来标定阈值。

代码原样搬自 `posture_app.py`。
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget)
from posture import review, settings
from posture.ui.widgets import ScaledImageLabel


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
        cfg = settings.load_config()      # 演示模式/引擎未起时，也能看已存的
        return review.ReviewStore(max_files=cfg.get("review_max_files", 300), enabled=True)

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
        cfg = settings.load_config()
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

        self.img = ScaledImageLabel("（没有待审核的现场）")
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
            self.img.clear_image()
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
                if pix.isNull():
                    # QPixmap 读失败不抛异常，只给个空对象 —— 显式当错误处理，
                    # 否则界面会"什么都没有"，看不出是图坏了
                    raise ValueError("QPixmap 为空")
                # 同预览：只交源图，缩放交给控件在绘制时做（见 ScaledImageLabel）
                self.img.set_image(pix)
            except Exception:
                # 先清源图再显示错误文字，否则旧图会盖在文字上
                self.img.clear_image()
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
            cfg = settings.load_config()
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
        cfg = settings.load_config()
        cfg["review_enabled"] = bool(self.chk_on.isChecked())
        cfg["review_sample_sec"] = sec
        cfg["review_keep_fullres"] = bool(self.chk_full.isChecked())
        if not settings.save_config(cfg):
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

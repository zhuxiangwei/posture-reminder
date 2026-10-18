# -*- coding: utf-8 -*-
"""界面基础控件：按尺寸自绘缩放的图片控件 + 手绘表情。

代码原样搬自 `posture_app.py`。
"""

from PySide6.QtCore import (Qt, QSize, QRectF, QPointF)
from PySide6.QtGui import (QBrush, QColor, QPainter, QPen)
from PySide6.QtWidgets import (QLabel, QWidget)
from posture.ui.theme import LEVEL_STYLE


class ScaledImageLabel(QLabel):
    """图片**始终**按当前控件尺寸铺满的 QLabel（HiDPI 安全）。

    为什么不用「setPixmap 前先缩放到控件尺寸」：
      那会把尺寸**固化在设置那一刻**。而本界面的预览区高度会变
      （表头提示语一行/两行不同 -> 布局回流），控件一变，
      固化的小 pixmap 就只占一角 —— 实测表现为「视频窗口只显示四分之一」，
      且**时好时坏**（取决于最后一次 setPixmap 时控件多大）。
      → 改成在 paintEvent 里按**当前**尺寸缩放，任何时刻都自洽，也不依赖调用时序。

    HiDPI 要点：缩放到「控件逻辑尺寸 × devicePixelRatio」，再把 dpr 设回去。
    直接 scaled(控件逻辑尺寸) 在 2x 屏上会只占 1/4 面积。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._src = None          # 源图（保持原始分辨率，缩放推迟到绘制时）

    def set_image(self, pix):
        """设置源图。传 None 等价于 clear_image()。"""
        self._src = None if (pix is None or pix.isNull()) else pix
        self.setText("")          # 有图就不显示占位文字
        self.update()

    def clear_image(self):
        self._src = None
        self.update()

    def paintEvent(self, ev):
        # 先让 QLabel 走默认绘制：保留样式表背景与占位文字
        super().paintEvent(ev)
        if self._src is None:
            return
        dpr = self.devicePixelRatioF() or 1.0
        sz = self.size()
        target = QSize(max(1, int(sz.width() * dpr)), max(1, int(sz.height() * dpr)))
        pm = self._src.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        # KeepAspectRatio 后按逻辑尺寸居中
        w = pm.width() / dpr
        h = pm.height() / dpr
        p = QPainter(self)
        p.drawPixmap(int((sz.width() - w) / 2), int((sz.height() - h) / 2), pm)
        p.end()


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

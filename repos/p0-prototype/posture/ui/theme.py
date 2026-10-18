# -*- coding: utf-8 -*-
"""界面样式常量与字体选择。

代码原样搬自 `posture_app.py`（配色、QSS、文案格式化都没改）。
配色原则见仓库根 `posture_app.py` 的模块文档：浅色、暖、不用红。
"""

from PySide6.QtGui import QFont


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
/* 摄像头状态横幅：底色/字色由代码按"降档(暖橙)"与"断线(灰蓝)"两种语气给，
   这里只固定形状。⚠️ 两种都不用红色 —— 那是惩罚感，孩子会想关掉程序。 */
#camWarn { border-radius: 10px; padding: 8px 12px; font-size: 12px; }
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

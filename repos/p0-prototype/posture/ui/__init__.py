# -*- coding: utf-8 -*-
"""坐姿小卫士 · 图形界面（PySide6 / Qt）

由 `posture_app.py` 拆分而来，代码是原样搬过来的，只改了 import 与模块名前缀
（`posture_monitor`/`posture_engine` → `posture.*`）。

依赖顺序（无环）：

    theme  ->  widgets  ->  review_panel  ->  parent_dialog  ->  main_window
                                                            ->  selftest -> app

    theme          LEVEL_STYLE / QSS / fmt_dur / apply_app_font
    widgets        ScaledImageLabel / FaceWidget
    review_panel   ReviewPanel
    parent_dialog  ParentDialog
    main_window    PostureApp
    selftest       _fake_snapshot / _StubEngine / selftest
    app            main()（argparse + QApplication 装配）

⚠️ 本包会真实导入 Qt，所以 `posture/__init__.py` 里绝对不碰它 ——
判定/引擎那些模块必须能在没装 PySide6 的机器上 import（tests/test_offline.py 就靠这点）。
设计原则与技术要点见仓库根 `posture_app.py` 的模块文档。
"""

from posture.ui.main_window import PostureApp
from posture.ui.parent_dialog import ParentDialog
from posture.ui.review_panel import ReviewPanel
from posture.ui.selftest import selftest
from posture.ui.theme import LEVEL_STYLE, QSS, apply_app_font, fmt_dur
from posture.ui.widgets import FaceWidget, ScaledImageLabel

__all__ = ["PostureApp", "ParentDialog", "ReviewPanel", "ScaledImageLabel",
           "FaceWidget", "selftest", "QSS", "LEVEL_STYLE", "fmt_dur",
           "apply_app_font"]

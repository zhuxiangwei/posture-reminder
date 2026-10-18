# -*- coding: utf-8 -*-
"""图形界面入口：argparse 参数 + QApplication 装配。

仓库根的 `posture_app.py` 只是薄壳启动器，真正的 `main()` 在这里。
"""

import argparse
import os
import sys

from PySide6.QtWidgets import QApplication
from posture.ui.main_window import PostureApp
from posture.ui.selftest import selftest
from posture.ui.theme import QSS, apply_app_font


# ---------------------------------------------------------------- 入口

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

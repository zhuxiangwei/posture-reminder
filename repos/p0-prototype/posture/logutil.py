# -*- coding: utf-8 -*-
"""运行日志落盘。

单独一个模块的原因：日志以前挂在 `posture_engine.RunLog`，
但 CLI（`posture.cli`）和自诊断（`posture_app --selftest`）同样需要它，
放在引擎里就逼着它们去 import 引擎（进而 import Qt/mediapipe）。
"""

import os
import time

from posture import paths


class RunLog:
    """把界面/引擎的运行日志同时写到磁盘。

    为什么要它：GUI 的「运行日志」页只活在内存里（上限 300 行），程序一关就没了 ——
    出了问题只能靠人复述"当时显示了什么"。留一份文件，复测/排障时直接把当天的
    `logs\\run-YYYYMMDD.log` 发过来，就能看到完整时间线。

    规则：
      · 默认 `logs/run-YYYYMMDD.log`（按天一个文件；`--log-file` 可改路径）
      · 只保留最近 KEEP_DAYS 天，启动时清一次
      · **任何异常都被吞掉** —— 记日志失败绝不能拖垮提醒主功能
      · `enabled=False`（`--no-log`）时完全不动磁盘
    """

    KEEP_DAYS = 7

    def __init__(self, path=None, enabled=True):
        self.enabled = bool(enabled)
        self.path = None
        if not self.enabled:
            return
        try:
            p = path or os.path.join(paths.LOG_DIR, time.strftime("run-%Y%m%d.log"))
            # 给的是目录也接受（末尾斜杠或已存在的目录）
            if p.endswith((os.sep, "/")) or os.path.isdir(p):
                p = os.path.join(p, time.strftime("run-%Y%m%d.log"))
            self.path = os.path.abspath(p)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n===== 启动 %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            self._prune()
        except Exception:
            # 盘写不了（只读目录/无权限）→ 静默降级成"不落盘"，不影响使用
            self.enabled, self.path = False, None

    def _prune(self):
        try:
            d = os.path.dirname(self.path)
            keep = {time.strftime("run-%Y%m%d.log",
                                  time.localtime(time.time() - i * 86400))
                    for i in range(self.KEEP_DAYS)}
            for name in os.listdir(d):
                if name.startswith("run-") and name.endswith(".log") and name not in keep:
                    os.remove(os.path.join(d, name))
        except Exception:
            pass

    def write(self, line):
        if not (self.enabled and self.path):
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

# -*- coding: utf-8 -*-
"""离屏界面自检（`--selftest`）：把每个状态分支都渲染一遍并强制绘制。

代码原样搬自 `posture_app.py`；摄像头状态那几项是重构后新增的（见 `_fake_cam`）。
"""

import time

from PySide6.QtCore import (Qt, QEvent)
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication
from posture import engine, review, settings, texts
from posture.ui.parent_dialog import ParentDialog


def _fake_cam(**kw):
    """假 `snapshot()["cam"]` —— 形状与 `engine.PostureEngine._cam_view` 输出一致。"""
    c = {"phase": "ready", "backend": "DSHOW", "camera": 0,
         "width": 3840, "height": 2160, "degraded": False,
         "reconnects": 0, "short": None, "stale_sec": 0.2,
         # 开流/重连期间真机才有值：此刻在试哪个组合（见 camera._note_trying）
         "trying": None}
    c.update(kw)
    return c


def _fake_snapshot(phase, **kw):
    s = {"phase": phase, "error": None, "preview": None, "kps": None,
         "level": "idle", "text": texts.TEXT_IDLE, "issue": None, "pending": None,
         "calib": {"done": 0, "need": 15, "sec_left": 0.0},
         "baseline": None, "detect_rate": None, "metrics": None,
         "stats": engine.PostureEngine._blank_stats(),
         # ★ 摄像头实况：重构后界面靠它画"开流进度 / 降档警告 / 重连状态"
         "cam": None, "cam_warnings": []}
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
        _fake_snapshot("opening", text=texts.TEXT_OPENING,
                       cam=_fake_cam(phase="opening")),
        _fake_snapshot("calibrating", calib={"done": 6, "need": 15, "sec_left": 4.0},
                       cam=_fake_cam()),
        _fake_snapshot("monitoring", level="good", text=texts.TEXT_GOOD,
                       baseline={"neck_angle": 3.0, "trunk_angle": None}, detect_rate=0.92,
                       cam=_fake_cam(),
                       stats={"total_sec": 600, "good_sec": 520, "good_ratio": 0.87,
                              "reminders": 3, "streak_sec": 130, "best_streak_sec": 490}),
        _fake_snapshot("monitoring", level="warn", text="头抬高一点点～",
                       pending=("头前倾", 2.4), detect_rate=0.88, cam=_fake_cam()),
        _fake_snapshot("monitoring", level="nodetect", text=texts.TEXT_NODETECT,
                       detect_rate=0.12, cam=_fake_cam()),
        # ★ 请求 4K 只拿到 1080p：界面必须显式说，不许静默降档
        _fake_snapshot("monitoring", level="good", text=texts.TEXT_GOOD,
                       cam=_fake_cam(width=1920, height=1080, degraded=True),
                       cam_warnings=["只拿到 1920x1080，低于请求的 3840x2160。"]),
        # ★ 采集中断：引擎给的 text 是「正在自动重连」，界面不得画成"看不到人"
        _fake_snapshot("monitoring", level="idle", text=texts.TEXT_RECONNECTING,
                       cam=_fake_cam(phase="reconnecting", stale_sec=9.0, reconnects=2)),
        _fake_snapshot("error", error="标定失败：只采到 3 个有效样本。",
                       error_detail="排查顺序：\n  1) 是否被别的程序占用"),
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
    orig_lb = settings.load_baseline
    settings.load_baseline = lambda cfg=None: {"neck_angle": 4.0, "trunk_angle": None,
                                         "sh_ear_len": 210.0}
    try:
        for swing, expect in ((3.0, "太正对孩子"), (7.0, "勉强可用"), (13.0, "机位没问题")):
            def _snap(neck):
                return _fake_snapshot("monitoring", level="good", text=texts.TEXT_GOOD,
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
        settings.load_baseline = orig_lb
        app.engine = None

    # 摄像头状态渲染：降档必须显式、断线必须有进度、正常时不许常驻警告
    # （这几条是重构新增的契约，只靠源码字符串锁不住 —— 直接渲染一遍最省事）
    app._render_snapshot(_fake_snapshot(
        "monitoring", level="good", text=texts.TEXT_GOOD,
        cam=_fake_cam(width=1920, height=1080, degraded=True),
        cam_warnings=["只拿到 1920x1080，低于请求的 3840x2160。"]))
    assert app.lbl_cam.isVisible(), "降档时横幅必须可见（不许静默降档）"
    assert "1920x1080" in app.lbl_cam.text(), app.lbl_cam.text()
    app.update()
    checks += 1

    app._render_snapshot(_fake_snapshot("monitoring", level="good", text=texts.TEXT_GOOD,
                                        cam=_fake_cam()))
    assert not app.lbl_cam.isVisible(), "档位正常时横幅必须隐藏，别常驻稀释注意力"
    checks += 1

    app._render_snapshot(_fake_snapshot(
        "monitoring", level="idle", text=texts.TEXT_RECONNECTING,
        cam=_fake_cam(phase="reconnecting", stale_sec=9.0, reconnects=2,
                      short="摄像头没画面了")))
    assert app.lbl_cam.isVisible() and "自动重连" in app.lbl_cam.text(), app.lbl_cam.text()
    assert "第 3 次" in app.lbl_cam.text(), app.lbl_cam.text()   # reconnects=2 -> 第 3 次尝试
    # 状态卡也必须说设备的事，不能说"请坐到座位中间"（那会把设备故障算到孩子头上）
    assert "自动重连" in app.lbl_sub.text(), app.lbl_sub.text()
    assert texts.TEXT_NODETECT not in app.lbl_big.text(), app.lbl_big.text()
    checks += 1

    app._cam_t0 = time.perf_counter() - 3.0      # 假装已经等了 3 秒
    # ① 正常路径：采集线程报了"正在试哪个组合"，界面必须把它说全
    app._render_snapshot(_fake_snapshot(
        "opening", text=texts.TEXT_OPENING,
        cam=_fake_cam(phase="opening", backend=None, width=None, height=None,
                      trying={"backend": "MSMF", "camera": 0, "width": 1280,
                              "height": 720, "attempt": 2})))
    sub = app.lbl_sub.text()
    assert "等了" in sub and "MSMF" in sub and "1280x720" in sub and "第 2 轮" in sub, sub
    assert "None" not in sub, f"缺字段时不许把 None 拼进话里：{sub}"
    assert not app.lbl_cam.isVisible(), "还没出结果时不要提前拉警告横幅"
    checks += 1

    # ② 退回路径：老快照没有 trying，界面也要有话可说（不许显示成 NonexNone）
    app._render_snapshot(_fake_snapshot("opening", text=texts.TEXT_OPENING,
                                        cam=_fake_cam(phase="opening")))
    assert "等了" in app.lbl_sub.text() and "DSHOW" in app.lbl_sub.text(), app.lbl_sub.text()
    assert "None" not in app.lbl_sub.text(), app.lbl_sub.text()
    app._render_snapshot(_fake_snapshot("opening", text=texts.TEXT_OPENING,
                                        cam=_fake_cam(phase="opening", backend=None,
                                                      width=None, height=None)))
    assert "None" not in app.lbl_sub.text(), \
        f"什么都不知道时宁可只说等了多久：{app.lbl_sub.text()}"
    checks += 1

    # 家长设置里的设备实况（家长要看的是**实际**档位）
    app.engine = _StubEngine(_fake_snapshot(
        "monitoring", level="good", cam=_fake_cam(width=1920, height=1080, degraded=True,
                                                  reconnects=3),
        cam_warnings=["只拿到 1920x1080，低于请求的 3840x2160。"]))
    dlg._refresh_live()
    assert "1920x1080" in dlg.cam_state.text(), dlg.cam_state.text()
    assert "低于请求档位" in dlg.cam_state.text(), dlg.cam_state.text()
    assert "重连 3 次" in dlg.cam_state.text(), dlg.cam_state.text()
    app.engine = None
    checks += 1

    # 开流还没成功时：这一行不许出现 None（档位是开出来那一刻才知道的）
    app.engine = _StubEngine(_fake_snapshot(
        "opening", text=texts.TEXT_OPENING,
        cam=_fake_cam(phase="opening", backend=None, camera=None,
                      width=None, height=None,
                      trying={"backend": "DSHOW", "camera": 0, "width": 3840,
                              "height": 2160, "attempt": 1})))
    dlg._refresh_live()
    cs = dlg.cam_state.text()
    assert "None" not in cs, f"开流途中家长页不许显示 None：{cs}"
    assert "正在试 DSHOW ／ 3840x2160" in cs, cs
    app.engine = None
    checks += 1

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
        st = review.ReviewStore(root=tmp, max_files=50, enabled=True)
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

        r = st.suggest_all(settings.load_config()).get("neck_angle")
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
    stub = _StubEngine(_fake_snapshot("monitoring", level="good", text=texts.TEXT_GOOD))
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

    print(f"界面自检通过：{checks} 项（9 个状态分支 + 预览渲染 + 家长设置窗口 "
          f"+ 机位判定 3 档 + 摄像头状态渲染 7 项（降档/正常/重连/开流进度×2/家长实况×2）"
          f"+ 审核面板 + 存图标注闭环 + 清空 + 坐正不发声 "
          f"+ 空格键暂停/继续 + 语音片段 {len(ParentDialog._available_voices())} 条）")
    app.close()

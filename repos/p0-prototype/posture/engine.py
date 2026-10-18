# -*- coding: utf-8 -*-
"""运行时引擎：采集 → 推理 → 判定 → 迟滞 → 语音提醒，跑在后台线程里。

界面只通过 `snapshot()` 无锁读状态、`pause()/resume()/recalibrate()/stop()` 下命令。

为什么必须分线程
----------------
单帧推理约 20ms、采样 5 FPS 看起来不重；但 `cap.read()` 会阻塞（驱动/曝光），
开流本身又要 1~15s，再加上标定要连续跑 10 秒 —— 这些放在 Qt 主线程里，
界面会在最关键的时刻卡死。所以**所有 I/O 与推理都不在 UI 线程**。

★ 本次重构在设备侧的结构变化（这是"加载很久/画面僵住"的真正解法）
------------------------------------------------------------------
重构前引擎线程**自己**开流、自己 `cap.read()`，于是：

    ① 开流的 5~20s 里界面只能干等，而且**没有任何进度**（用户视角 = 卡死）；
    ② `cap.read()` 一旦在 C 层阻塞，整条推理循环跟着僵，界面随之假死；
    ③ 摄像头被别的程序抢走 / USB 掉线后**没有恢复路径** ——
       画面停在最后一帧，程序看起来还在跑，实际早就不更新了。

现在：设备归 `camera.CameraStream`（自己的长命线程，创建与读取同线程），
引擎只做"取最新帧 → 推理 → 判定"，并从 `stream.status()` 实时向前台播报
opening / ready / reconnecting / failed。另外两条顺手的收益：

    · **模型构造与开流并行**：`create_from_options` 要 0.3~1s 且完全不碰摄像头，
      串行执行等于白等；
    · **基线绑定"实际"分辨率**（不是请求值）：请求 4K 却只拿到 1080p 时，
      用 4K 标的基线必须作废重标 —— 否则像素类指标 `sh_ear_len` 整体放大，
      「低头」判据永不成立，属于**静默漏报**。

对外契约
--------
    eng = PostureEngine(cfg)
    eng.start()
    snap = eng.snapshot()          # 任意时刻可读，返回一份浅拷贝
    eng.pause() / resume() / recalibrate() / stop()

snapshot() 的关键字段
---------------------
    phase    : idle|opening|calibrating|monitoring|error|stopped
    error    : str|None           短标题（界面直接显示）
    error_detail : str|None       展开才看的排查清单
    preview  : ndarray|None       缩小的 BGR 帧，给界面显示
    level    : good|warn|nodetect|idle   界面按它上色
    text     : str                给孩子看的一句话
    issue    : str|None           当前确认的问题（做提醒/统计用）
    pending  : (str, float)|None  观察中的问题与已持续秒数（可做倒计时）
    stats    : dict               今日统计
    calib    : dict               标定进度 {done, need, sec_left}
    cam      : dict               ★ 摄像头实况（见 `_cam_view`）——
                                   界面靠它显示开流进度、重连状态与降档警告
    cam_warnings : list[str]      ★ 非空就必须显示（低于请求档 / 没协商到 MJPG）
"""

import threading
import time
import traceback

from posture import camera, deps, gate, metrics as mx, pose, settings, texts
from posture.logutil import RunLog
from posture.paths import DEFAULT_MODEL, find_model
from posture.review import ReviewStore
from posture.voice import Speaker

# 兼容旧引用名（界面与测试里还有 `engine.TEXT_GOOD` 这类写法）
TEXT_GOOD = texts.TEXT_GOOD
TEXT_NODETECT = texts.TEXT_NODETECT
TEXT_PAUSED = texts.TEXT_PAUSED
TEXT_IDLE = texts.TEXT_IDLE
ISSUE_TIP = texts.ISSUE_TIP

# 等摄像头时的**外部**上限：开流自己有 budget，这里只多留一点余量给
# 非首选后端的子进程探测（首次运行最多 8s×2，之后落盘缓存就不再付）。
CAM_WAIT_EXTRA_SEC = 24.0


class PostureEngine(threading.Thread):
    def __init__(self, cfg=None, on_log=None):
        super().__init__(daemon=True, name="posture-engine")
        self.cfg = cfg or settings.load_config()
        self.on_log = on_log or (lambda msg: None)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._recalib = threading.Event()

        self._speaker = Speaker(mute=self.cfg.get("mute", False),
                                volume=self.cfg.get("volume", "中"),
                                on_log=self._log)
        self.review = ReviewStore(
            max_files=self.cfg.get("review_max_files", 300),
            enabled=self.cfg.get("review_enabled", True))

        # ★ 设备不在引擎手里：cap 归 CameraStream 的线程创建并持有。
        #   引擎只留一个引用，且**绝不**从本线程 release 它（见 CameraStream.stop 的文档）。
        self._stream = None
        self._lm = None
        self._gate = None
        self._base = None

        # self._eff = **实际**采集条件（索引/分辨率以开流结果为准）。
        # 基线的校验与落盘都用它，而不是用户请求值 —— 见 `_adopt_camera`。
        self._eff = dict(self.cfg)
        self._last_seq = 0          # 已消费到的帧号（避免同一帧推理两次）
        self._reconnects = 0        # 已发生的重连次数，用来发现"档位变了"

        self._snap = {
            "phase": "idle", "error": None, "error_detail": None,
            "preview": None, "kps": None,
            "level": "idle", "text": TEXT_IDLE, "issue": None, "pending": None,
            "calib": {"done": 0, "need": 15, "sec_left": 0.0},
            "baseline": None, "detect_rate": None, "metrics": None,
            "mute": bool(self.cfg.get("mute", False)),
            "volume": self.cfg.get("volume", "中"),
            "stats": self._blank_stats(),
            "cam": None, "cam_warnings": [],
        }

    # ---------------- 状态

    @staticmethod
    def _blank_stats():
        return {"total_sec": 0.0, "good_sec": 0.0, "good_ratio": None,
                "reminders": 0, "streak_sec": 0.0, "best_streak_sec": 0.0}

    def _set(self, **kw):
        with self._lock:
            self._snap.update(kw)

    def snapshot(self):
        with self._lock:
            s = dict(self._snap)
            s["stats"] = dict(s["stats"])
            s["calib"] = dict(s["calib"])
            s["cam_warnings"] = list(s["cam_warnings"])
            if s["cam"] is not None:
                s["cam"] = dict(s["cam"])
            return s

    def _log(self, msg):
        try:
            self.on_log(msg)
        except Exception:
            pass

    # ---------------- 指令

    def stop(self):
        self._stop.set()
        self._paused.clear()
        # 只发停止请求；真正的 release 由采集线程自己做（COM 绑定，别的线程不许碰）
        if self._stream is not None:
            self._stream.stop()

    def pause(self):
        self._paused.set()
        if self._gate:
            self._gate.reset()
        self._set(level="idle", text=TEXT_PAUSED, issue=None, pending=None)

    def resume(self):
        self._paused.clear()
        if self._gate:
            self._gate.reset()

    def recalibrate(self):
        self._recalib.set()

    def set_audio(self, mute=None, volume=None):
        with self._lock:
            if mute is not None:
                self._speaker.mute = bool(mute)
                self._snap["mute"] = bool(mute)
            if volume in self._speaker.GAIN:
                self._speaker.volume = volume
                self._snap["volume"] = volume

    def preview_voice(self, issue, volume=None):
        """试听一条提醒语。

        ⚠️ 刻意**忽略静音开关**：点了试听却没声音，家长会以为功能坏了。
        ⚠️ 返回值只表示"已排入播放队列"（播放是异步的），
           **不要**用它判断"文件是否存在" —— 播放线程会异步汇报真实结果。
        """
        sp = self._speaker
        old_mute, old_vol = sp.mute, sp.volume
        sp.mute = False
        if volume in sp.GAIN:
            sp.volume = volume
        try:
            return sp.say(issue)
        finally:
            sp.mute, sp.volume = old_mute, old_vol

    # ---------------- 摄像头进度

    def _cam_view(self, st):
        """把 `CameraStream.status()` 收成界面要的那几项。

        为什么要转一道：stream 的 dict 里有 `result`（一整串 summary 文本）和
        `stale_sec` 这类排查字段，界面每次刷新都拷一遍没必要；
        而 `phase / degraded / width / height / backend / camera / reconnects / short /
        trying` 这几个是**状态条与警告横幅**真正要的。
        `trying` = 开流期间"此刻正在试哪个组合"，只有 opening/reconnecting 有值。
        """
        return {k: st.get(k) for k in ("phase", "backend", "camera", "width", "height",
                                       "degraded", "reconnects", "short", "stale_sec",
                                       "trying")}

    def _on_cam_event(self, msg):
        """采集线程的进度播报 —— 直接进日志，界面用 `cam` 字段自己画进度。

        ⚠️ 这里**不**碰 `self._snap`（除只读外的状态由引擎线程自己刷）：
           采集线程与引擎线程是两个线程，界面读的是引擎的快照，
           让"谁维护快照"保持单一归属，省掉一类难查的竞态。
        """
        self._log(msg)

    def _adopt_camera(self, res):
        """开流（或重连）成功后，把**实际**采集条件写进 `_eff` 与快照。"""
        self._eff = dict(self.cfg)
        self._eff["camera"] = res.camera
        self._eff["width"] = res.width
        self._eff["height"] = res.height
        self._eff["model"] = self.cfg.get("model") or DEFAULT_MODEL
        self._set(cam=self._cam_view(self._stream.status()),
                  cam_warnings=list(res.warnings))
        # ⚠️ 故意**不**把降档结果写回 config.json：
        #    用户口径是"坚持 4K 优先"，config 里留请求值才保证下次仍然先试 4K。
        #    实际档位只影响"这份基线还能不能用"，不影响下次的野心。
        for w in res.warnings:
            self._log(f"[摄像头][警告] {w}")

    def _await_camera(self):
        """等采集线程把流开起来；期间把进度写进快照，界面能显示"已等几秒"。

        返回 OpenResult；失败抛 `camera.CameraError`（由 run() 兜成界面错误）。
        """
        limit = self._stream.budget + CAM_WAIT_EXTRA_SEC
        t0 = time.perf_counter()
        while not self._stop.is_set():
            st = self._stream.status()
            # ★ 把"正在试哪个后端/哪一档"实时递到界面（重构前这里什么都不写，
            #   开流的 1~15s 里界面只有一个"正在打开摄像头…"，用户视角等于卡死）。
            #   只读 `status()`、不改任何设备状态：快照的归属仍是引擎线程。
            self._set(cam=self._cam_view(st))
            ph = st["phase"]
            if ph == "ready":
                return self._stream.result
            if ph == "failed":
                raise camera.CameraError(st.get("short") or "摄像头开流失败",
                                         st.get("detail") or "")
            if ph == "stopped":
                return None
            if time.perf_counter() - t0 > limit:
                raise camera.CameraError(
                    f"摄像头开流超时（超过 {limit:.0f}s 仍没有画面）",
                    "采集线程还在重试。常见原因：设备被别的程序占用、"
                    "隐私设置未放行桌面应用、USB 供电不足。\n"
                    + camera.CHECKLIST)
            self._stop.wait(0.05)
        return None

    # ---------------- 线程主体

    def run(self):
        try:
            settings.apply_config(self.cfg)
            model = find_model(self.cfg.get("model") or DEFAULT_MODEL)
            self._set(phase="opening", text=texts.TEXT_OPENING,
                      error=None, error_detail=None, cam=None, cam_warnings=[])
            self._log(f"模型：{model}")
            # ⚠️ GUI 路径不经过 CLI 入口，那里的依赖守卫在这里不会自动执行。
            #    不补这一句，mediapipe 导入失败（例如本机 SAC 拦了 matplotlib 的
            #    _image.pyd）会一路走到 make_landmarker，报出与真病因毫无关系的
            #    `AttributeError: 'NoneType' object has no attribute 'PoseLandmarker'`。
            deps.need_mp()
            # ⚠️ 时间戳基准必须**全生命周期只设一次**。
            # MediaPipe 的 VIDEO 模式要求 detect_for_video 的时间戳单调递增；
            # 若在标定里用 `now - t0` 而重标定时把 t0 归零，时间戳会倒退，推理结果直接错。
            self._t0 = time.perf_counter()

            # ① 采集线程先起来（开流是耗时大头，让它和下面的模型构造并行）
            self._stream = camera.CameraStream(
                idx=self.cfg.get("camera"), backend=self.cfg.get("backend"),
                width=self.cfg.get("width"), height=self.cfg.get("height"),
                on_event=self._on_cam_event)
            self._stream.start()

            # ② 与此同时在本线程建模型（不碰设备，安全）
            lm = pose.make_landmarker(model)
            # ③ 等摄像头
            res = self._await_camera()
            if res is None:              # 停止请求
                return
            self._lm = lm
            self._log(f"摄像头就绪：{res.summary()}")
            self._adopt_camera(res)

            self._gate = gate.ReminderGate(self.cfg.get("sustain_sec"),
                                           self.cfg.get("clear_sec"),
                                           self.cfg.get("remind_gap_sec"),
                                           self.cfg.get("remind_gap_any_sec"))
            self._base = settings.load_baseline(self._eff)
            if self._base:
                self._log("已读回上次的基线（机位与分辨率都没变，无需重标）")
                self._set(baseline=self._base)
            else:
                why = settings.baseline_reason(self._eff)
                if why:
                    self._log(f"需要重新标定：{why}")
                if not self._calibrate():
                    return   # 标定失败已写入 error
            self._monitor()
        except camera.CameraError as e:
            # ★ 摄像头错误分两段给：短文案进 `error`（界面直接显示），
            #   完整排查清单进 `error_detail`（点开"详情"才看）——
            #   重构前是一大坨文本糊在弹窗里，家长根本看不懂该做什么。
            self._set(phase="error", error=e.short, error_detail=e.detail)
            self._log(f"启动失败：{e.short}\n{e.detail}")
        except SystemExit as e:
            # find_model / 依赖守卫用 sys.exit 报错，在子线程里会静默退出线程
            self._set(phase="error", error=str(e) or "启动失败")
            self._log(f"启动失败：{e}")
        except Exception as e:
            # ⚠️ 关闭期间的解释器拆除噪音，不要当成"故障"记进日志。
            #    实测关窗时若引擎正卡在 detect_for_video 里，mediapipe 的
            #    ThreadPoolExecutor 已被 atexit 关掉，会抛
            #        RuntimeError: cannot schedule new futures after shutdown
            #    这不是功能问题（进程正常退出），但记成"异常"会往日志里写一段
            #    看不懂的栈、误导后续排查。
            if self._stop.is_set():
                self._log(f"（关闭中，忽略：{type(e).__name__}: {e}）")
            else:
                self._set(phase="error", error=f"{type(e).__name__}: {e}",
                          error_detail=traceback.format_exc())
                self._log("异常：\n" + traceback.format_exc())
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
            if self._lm is not None:
                try:
                    self._lm.close()
                except Exception:
                    pass
            with self._lock:
                if self._snap["phase"] != "error":
                    self._snap["phase"] = "stopped"

    # ---------------- 取帧

    def _sample(self, t_sec):
        """从采集线程取**最新一帧**并推理。

        返回 (frame, metrics|None, kps|None, cam_state)：
          · frame 为 None = 这一拍没有新帧（设备停摆/重连中），调用方**不要**判定，
            也不要算进统计 —— 否则"摄像头被抢了"会被统计成"孩子一直坐姿差"。
          · cam_state 是 `CameraStream.status()["phase"]`，界面用它区分
            "没人"（nodetect）与"没画面"（reconnecting/failed）。

        ⚠️ 与旧的 `grab(cap, ...)` 的区别：不再在本线程 `cap.read()`。
           引擎拿到的帧永远是"此刻最新"，而 read 阻塞只发生在采集线程里，
           界面与判定循环不会被设备拖死。
        """
        st = self._stream.status()
        frame, seq = self._stream.latest()
        if frame is None or seq == self._last_seq or st["phase"] not in ("ready",):
            return None, None, None, st["phase"]
        self._last_seq = seq
        m, kps = pose.detect(self._lm, frame, t_sec, mx.compute_side_metrics)
        return frame, m, kps, st["phase"]

    def _camera_moved(self):
        """重连之后**实际档位可能变了** —— 必须重新校验基线，否则像素类判据会静默失效。

        典型场景：4K 标好的基线，掉线后驱动只给了 1080p。
        此时 `sh_ear_len` 整体缩小一半，「低头」的收缩判据永不成立 = 漏报。
        """
        st = self._stream.status()
        n = int(st.get("reconnects") or 0)
        if n == self._reconnects:
            return False
        self._reconnects = n
        res = self._stream.result
        if res is None:
            return False
        if (res.width, res.height, res.camera) == (self._eff.get("width"),
                                                   self._eff.get("height"),
                                                   self._eff.get("camera")):
            return False
        self._log(f"[重连] 采集条件变了："
                  f"{self._eff.get('width')}x{self._eff.get('height')}"
                  f" 索引 {self._eff.get('camera')} -> "
                  f"{res.width}x{res.height} 索引 {res.camera}，基线要重新校验")
        self._adopt_camera(res)
        return True

    # ---------------- 标定

    def _calibrate(self):
        need_n = 15
        calib_sec = float(self.cfg.get("calib_sec", 10.0))
        self._set(phase="calibrating", level="idle",
                  text="请坐正，正在学习你的坐姿…",
                  calib={"done": 0, "need": need_n, "sec_left": calib_sec})
        self._log(f"开始标定：端坐保持 {calib_sec:.0f} 秒")

        interval = 1.0 / max(1, int(self.cfg.get("fps_sample", settings.FPS_SAMPLE)))
        samples, last = [], None
        t0 = time.perf_counter()          # 标定计时用（重标时归零）
        next_run = t0
        hard_limit = max(calib_sec * 4.0, 60.0)

        while not self._stop.is_set():
            now = time.perf_counter()
            if self._recalib.is_set():
                self._recalib.clear()
                samples, t0 = [], now      # 只重置标定计时，不动 self._t0
            if now >= next_run:
                next_run = now + interval
                fr, m, kps, cam_phase = self._sample(now - self._t0)
                if cam_phase == "failed":
                    self._set(phase="error", error=texts.TEXT_CAM_FAILED,
                              error_detail=self._stream.status().get("detail"))
                    return False
                if fr is not None:
                    last = pose.shrink(fr)
                if m:
                    samples.append(m)
                self._set(preview=last, kps=kps if fr is not None else None,
                          metrics=m, cam=self._cam_view(self._stream.status()),
                          calib={"done": len(samples), "need": need_n,
                                 "sec_left": max(0.0, calib_sec - (now - t0))})
            if len(samples) >= need_n and now - t0 > calib_sec:
                break
            if now - t0 > hard_limit:
                break
            time.sleep(0.005)

        if self._stop.is_set():
            return False
        if len(samples) < 10:
            st = self._stream.status()
            extra = ""
            if st["phase"] != "ready":
                extra = (f"\n\n另外：标定期间摄像头状态是「{st['phase']}」，"
                         "先把设备问题解决（是否被占用 / 是否掉线），再重新标定。")
            self._set(phase="error",
                      error=f"标定失败：只采到 {len(samples)} 个有效样本。\n"
                            "请确认摄像头拍到了完整的上半身、光线充足，然后重试。"
                            + extra)
            return False

        self._base = mx.build_baseline(samples)
        settings.save_baseline(self._eff, self._base)
        self._set(baseline=self._base,
                  calib={"done": len(samples), "need": need_n, "sec_left": 0.0})
        self._log("标定完成：" + ", ".join(f"{k}={v:.3f}"
                                          for k, v in sorted(self._base.items())))
        self._speaker.chime("done")
        return True

    # ---------------- 监测

    def _monitor(self):
        interval = 1.0 / max(1, int(self.cfg.get("fps_sample", settings.FPS_SAMPLE)))
        self._set(phase="monitoring", level="idle", text=TEXT_IDLE)
        self._log(f"进入监测（{1.0/interval:.0f} FPS 采样）")

        stats = self._blank_stats()
        t_prev = time.perf_counter()
        next_run = t_prev
        hit_n = tot_n = 0
        detect_report_at = t_prev + 20.0
        sample_sec = float(self.cfg.get("review_sample_sec", 180.0) or 0)
        next_sample_at = t_prev + sample_sec
        offline_since = None               # 本次"没有画面"从什么时候开始

        while not self._stop.is_set():
            now = time.perf_counter()

            # ---------- 设备侧优先：重连后档位可能变了
            if self._camera_moved():
                self._base = settings.load_baseline(self._eff)
                if not self._base:
                    self._log("[重连] 采集条件与基线不匹配，需要重新标定")
                    self._set(phase="calibrating")
                    if not self._calibrate():
                        return
                    self._set(phase="monitoring")
                    stats = self._blank_stats()
                    t_prev = time.perf_counter()
                    continue

            if self._recalib.is_set():
                self._recalib.clear()
                self._set(phase="calibrating")
                if not self._calibrate():
                    return
                self._set(phase="monitoring")
                stats = self._blank_stats()
                t_prev = time.perf_counter()
                continue

            if self._paused.is_set():
                time.sleep(0.1)
                t_prev = time.perf_counter()
                continue

            if now < next_run:
                time.sleep(0.004)
                continue
            next_run = now + interval
            dt = max(0.0, now - t_prev)
            t_prev = now

            fr, m, _kps, cam_phase = self._sample(now - self._t0)
            st = self._stream.status()

            if cam_phase != "ready":
                # ---------- 没有画面（掉线 / 重连中 / 失败）
                # ⚠️ 这一拍**不参与统计**，也不判"没检测到人"：
                #    重构前这里什么都不同步，界面就表现为"画面静止但程序还在跑" ——
                #    孩子其实没被提醒，家长以为一切正常。
                if offline_since is None:
                    offline_since = now
                    self._log(f"[设备] 采不到画面（状态={cam_phase}），"
                              "判定暂停、统计暂停，等待自动重连")
                txt = (texts.TEXT_RECONNECTING if cam_phase == "reconnecting"
                       else texts.TEXT_CAM_FAILED)
                self._set(cam=self._cam_view(st), level="idle", text=txt,
                          issue=None, pending=None, stats=stats,
                          # 预览保留最后一帧（清屏会让人以为程序也挂了）
                          kps=None, metrics=None)
                continue
            if offline_since is not None:
                self._log(f"[设备] 已恢复画面（中断 {now - offline_since:.0f}s）")
                offline_since = None
                self._set(cam=self._cam_view(st), cam_warnings=list(st["warnings"]))

            prev = pose.shrink(fr) if fr is not None else None

            if fr is None:
                # 设备正常但这一拍没有新帧（低帧率设备 + 高采样）：什么都不做，
                # 尤其**不要**把它算成"没检测到人"。
                continue

            if m is None:
                # ⚠️ 没检测到人**不参与统计**，否则孩子一离座就被算成"坐姿差"。
                tot_n += 1
                self._set(preview=prev, kps=None, metrics=None, level="nodetect",
                          text=TEXT_NODETECT, issue=None, pending=None, stats=stats)
            else:
                tot_n += 1
                hit_n += 1
                issues = mx.judge_side(m, self._base)
                stg = self._gate.update(issues, now)

                stats["total_sec"] += dt
                if issues:
                    stats["streak_sec"] = 0.0
                else:
                    stats["good_sec"] += dt
                    stats["streak_sec"] += dt
                    stats["best_streak_sec"] = max(stats["best_streak_sec"],
                                                   stats["streak_sec"])
                stats["good_ratio"] = (stats["good_sec"] / stats["total_sec"]
                                       if stats["total_sec"] > 1e-6 else None)

                if stg.fire:
                    stats["reminders"] += 1
                    queued = self._speaker.say(stg.fire)
                    self._log(f"提醒：{stg.fire}"
                              f"（{'已排入播放队列' if queued else '静音中，未播放'}）")
                    # 存现场图供家长审核 —— 这是拿到"误报率"验收指标的唯一现实途径
                    if self.review.save(prev, m, self._base, stg.fire,
                                        kind="reminder", cfg=self._eff, full_frame=fr):
                        self._log(f"已存现场图待审核（{stg.fire}）：家长设置 → 审核判定")

                # ⚠️ 坐正了**刻意不发声**（需求原话：「坐姿没问题就不要发出任何声音」）。
                #    早期这里会 chime("good") 给个正反馈提示音 —— 已移除。
                #    正向激励改用界面表达（绿色大字「坐得真棒！」+ 连续良好计时），
                #    声音只留给"需要纠正"这一种情况。

                if stg.issue:
                    tip = texts.ISSUE_TIP.get(stg.issue, stg.issue)
                    pending = (stg.issue, stg.elapsed) if not stg.confirmed else None
                    self._set(preview=prev, kps=None, metrics=m, level="warn",
                              text=tip, issue=stg.issue, pending=pending, stats=stats)
                else:
                    self._set(preview=prev, kps=None, metrics=m, level="good",
                              text=TEXT_GOOD, issue=None, pending=None, stats=stats)
                    # 定期存一张"判为良好"的抽样。
                    # ⚠️ 这不是多余的：漏报时程序不会存任何图，
                    #    没有这类抽样，漏报率就永远测不出来。
                    if sample_sec > 0 and now >= next_sample_at:
                        next_sample_at = now + sample_sec
                        if self.review.save(prev, m, self._base, None,
                                            kind="sample", cfg=self._eff):
                            self._log("已存一张抽样现场图（判为良好，供家长查漏报）")

            if now >= detect_report_at:
                detect_report_at = now + 30.0
                rate = hit_n / max(1, tot_n)
                self._set(detect_rate=rate)
                if rate < 0.3 and tot_n > 30:
                    self._log(f"⚠️ 检出率仅 {rate*100:.0f}%——摄像头可能没拍到人，"
                              "或光线太暗/机位不对。检出率低时判定结果不可信。")

        self._log("监测已停止")


# 兼容旧引用名：老脚本里 `pe.RunLog` / `pe.ReviewStore` / `pe.Speaker` 还能用
# （界面自己已改用 `from posture import logutil` 直调，不再经过这里）
__all__ = ["PostureEngine", "RunLog", "ReviewStore", "Speaker"]

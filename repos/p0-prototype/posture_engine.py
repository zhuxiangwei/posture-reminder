#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐姿提醒 · 运行时引擎（线程化）
================================

把「采集 → 推理 → 判定 → 迟滞 → 语音提醒」整条链路封成一个后台线程，
图形界面只通过 `snapshot()` 无锁读取状态、`pause()/resume()/recalibrate()/stop()` 下命令。

为什么必须分线程
----------------
单帧推理约 20ms、采样 5 FPS，看起来不重；但 `cap.read()` 会阻塞（驱动/曝光），
再加上标定阶段要连续跑 10 秒——这些放在 Tk 主线程里，界面会在最关键的时刻卡死。
所以**所有 I/O 与推理都在本线程，UI 线程只画界面**。

对外契约
--------
    eng = PostureEngine(cfg)
    eng.start()                      # 起线程（daemon，进程退出自动收）
    snap = eng.snapshot()            # 任意时刻可读，返回一份浅拷贝
    eng.pause() / eng.resume() / eng.recalibrate() / eng.stop()

snapshot() 的关键字段
---------------------
    phase    : idle|opening|calibrating|monitoring|error|stopped
    error    : str|None           出错原因（GUI 要弹给人看）
    preview  : ndarray|None       缩小的 BGR 帧，给界面显示
    level    : good|warn|nodetect|idle   界面按它上色
    text     : str                给孩子看的一句话
    issue    : str|None           当前确认的问题（做提醒/统计用）
    pending  : (str, float)|None  观察中的问题与已持续秒数（可做倒计时）
    stats    : dict               今日统计
    calib    : dict               标定进度 {done, need, sec_left}
"""

import json
import os
import queue
import threading
import time
import traceback

import posture_monitor as pm

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
BASELINE_PATH = os.path.join(HERE, "baseline.json")

DEFAULT_CONFIG = {
    "camera": 0,
    "side_only": True,
    "side_yaw": 90.0,
    "model": pm.DEFAULT_MODEL,
    "width": pm.DEFAULT_CAM_W,
    "height": pm.DEFAULT_CAM_H,
    "fps_sample": pm.FPS_SAMPLE,
    "calib_sec": 10.0,
    "mute": False,
    "volume": "中",             # 低 / 中 / 高
    "sustain_sec": pm.SUSTAIN_SEC,
    "clear_sec": pm.CLEAR_SEC,
    "remind_gap_sec": pm.REMIND_GAP_SEC,
    "remind_gap_any_sec": pm.REMIND_GAP_ANY_SEC,
    "side_thresholds": dict(pm.SIDE_THRESHOLDS),
    "thresholds": dict(pm.THRESHOLDS),
    # 判定现场留存（供家长审核 —— 这是拿到「误报率 <10%」验收指标的唯一现实途径）
    "review_enabled": True,        # 全部本地存储、不上传；家长可一键清空
    "review_max_files": 300,       # 超出上限自动删最旧的
    "review_sample_sec": 180.0,    # 每隔多久另存一张"判为良好"的抽样（否则测不出漏报）
    # 2026-09-26 用户改口径：摄像头换了高清、并要求**处处都用最高清档**，
    # 所以「提醒现场」的原图**默认就存**（原来为了省空间是关的）。
    # 代价：1080p 原图每张约 150~500KB；且 `_full.jpg` 也计入 review_max_files，
    #       所以实际能留存的**条数**会变少（想省空间就在「家长设置 → 审核判定」关掉）。
    # 剪枝/清空会把原图和缩略图一起处理（见 _files_of），不会留孤儿文件。
    "review_keep_fullres": True,
}

# 给孩子看的文案：温和、非命令式，而且**不用红色**。
# 报告 §8：正向激励优先，避免惩罚性语气——叫得太凶，孩子三天就把程序关了。
ISSUE_TIP = {
    "头前倾": "头抬高一点点～",
    "低头": "离本子远一点点～",
    "塌腰": "背挺直一点～",
    "歪头": "头摆正一点～",
    "歪肩": "肩膀放平～",
    "太近": "往后靠一点点～",
    "含胸驼背": "挺起胸膛来～",
}
TEXT_GOOD = "坐得真棒！"
TEXT_NODETECT = "看不到你啦，坐过来一点"
TEXT_PAUSED = "先休息一下"
TEXT_IDLE = "准备好了就点开始"


# ---------------------------------------------------------------- 配置 / 基线

def load_config(path=CONFIG_PATH):
    cfg = dict(DEFAULT_CONFIG)
    cfg["side_thresholds"] = dict(DEFAULT_CONFIG["side_thresholds"])
    cfg["thresholds"] = dict(DEFAULT_CONFIG["thresholds"])
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k, v in saved.items():
                if k in ("side_thresholds", "thresholds") and isinstance(v, dict):
                    cfg[k].update({kk: float(vv) for kk, vv in v.items()})
                else:
                    cfg[k] = v
        except Exception:
            pass    # 配置文件坏了就用默认值，不能因此起不来
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def apply_config(cfg):
    """把配置灌进 posture_monitor 的模块级参数。

    ⚠️ 判定函数（judge_side 等）读的是模块全局，这是既有设计。
    测试直接调判定函数时用的就是默认值，所以这里做的是"运行时覆盖"，
    而不是改签名——改签名会把 60 个回归用例全部牵动。
    """
    pm.SIDE_YAW_DEG = float(cfg.get("side_yaw", 90.0))
    pm.FPS_SAMPLE = int(cfg.get("fps_sample", pm.FPS_SAMPLE))
    for k, v in cfg.get("side_thresholds", {}).items():
        pm.SIDE_THRESHOLDS[k] = float(v)
    for k, v in cfg.get("thresholds", {}).items():
        pm.THRESHOLDS[k] = float(v)


def load_baseline(cfg, path=BASELINE_PATH):
    """读回上次的基线。机位/偏航角/分辨率变了就作废（基线是跟着机位走的）。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if d.get("camera") != cfg.get("camera") or \
       abs(float(d.get("side_yaw", -1)) - float(cfg.get("side_yaw", 0))) > 1e-6:
        return None
    return d.get("base") or None


def save_baseline(cfg, base, path=BASELINE_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"camera": cfg.get("camera"), "side_yaw": cfg.get("side_yaw"),
                       "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "base": base}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- 语音

class Speaker:
    """语音播放。支持静音与音量档位。

    ⚠️ 播放**跑在独立线程里**，`say()` 只入队后立刻返回。两个理由：

    1. **MCI 的 open/close 是阻塞调用**。原来它跑在推理线程里，
       一播语音就卡住采集循环 —— 实测表现为"识别结果出来时卡一下"。
    2. **队列深度只有 1，新提醒顶掉还没播的旧提醒**。提醒有时效性，
       串行排队会让语音越拖越晚、最后完全对不上现场；积压一串过时语音比丢掉更糟。

    ⚠️ 因此 `say()` 的返回值只表示"**已排入队列**"，不代表播放成功。
       真实结果通过 on_log 回调异步汇报（**含失败原因**）——
       原来只报"已播放/退回提示音"，出错时什么线索都没有，没法查。

    两套播放通道：

    | 通道 | 支持 | 音量 | 用在哪 |
    |---|---|---|---|
    | **winsound** | **仅 wav** | ❌（用 numpy 缩放 PCM 振幅代替） | **首选** |
    | MCI（winmm，ctypes） | mp3 + wav | ✅ `setaudio volume 0~1000` | mp3 只能走它 |

    ⚠️ **为什么 wav 优先、winsound 优先**：
      `PlaySound(SND_ASYNC)` 是**真异步**、没有设备 open/close 开销；
      而 MCI 的 open/close 是**阻塞调用**，且别名全局唯一 ——
      新播必须先 `close`，会把**正在播的那条掐断**（实测：听感就是"有的没播出来"）。
      所以语音统一生成 wav（见 tools/make_voice.py），MCI 只作 mp3 的兜底。

    文件优先级：`<问题名>.wav` > `<问题名>.mp3`。
    """

    GAIN = {"低": 0.40, "中": 0.70, "高": 1.00}
    _ALIAS = "posture_voice"

    def __init__(self, voice_dir=None, mute=False, volume="中", on_log=None):
        self.voice_dir = voice_dir or pm.VOICE_DIR
        self.mute = bool(mute)
        self.volume = volume if volume in self.GAIN else "中"
        self.last_error = None
        self._on_log = on_log or (lambda msg: None)
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, name="voice", daemon=True)
        self._t.start()

    def _log(self, msg):
        try:
            self._on_log(msg)
        except Exception:
            pass

    # ---------------- 播放线程

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            kind, arg = item
            try:
                if kind == "file":
                    self._play_file(arg)
                    self._log(f"🔊 播放 {os.path.basename(arg)}")
                elif kind == "chime":
                    import winsound
                    winsound.Beep(int(arg), 120)
                else:
                    self._beep()
            except Exception as e:
                self.last_error = f"{kind}: {e}"
                name = os.path.basename(arg) if kind == "file" else kind
                self._log(f"⚠️ 播放失败（{name}）：{e}")

    def _play_file(self, path):
        """按扩展名决定通道顺序。

        - `.wav` → **winsound 优先**（真异步、无设备开关开销），MCI 兜底
        - `.mp3` → 只能 MCI（winsound 不支持 mp3）
        """
        if path.lower().endswith(".wav"):
            chain = (("winsound", self._say_winsound), ("MCI", self._say_mci))
        else:
            chain = (("MCI", self._say_mci), ("winsound", self._say_winsound))
        for name, fn in chain:
            try:
                fn(path)
                return
            except Exception as e:
                self.last_error = f"{name}: {e}"
        raise OSError(self.last_error or "两条通道都失败")

    def _offer(self, item):
        """入队；满了就丢掉旧的（提醒有时效性，不该排队等待）。"""
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
                return True
            except queue.Full:
                return False

    def stop(self):
        self._stop.set()

    # ---------------- 文件定位

    def _find(self, issue):
        """wav 优先于 mp3 —— wav 能走 winsound，播放更稳（见类文档）。"""
        for ext in (".wav", ".mp3"):
            p = os.path.join(self.voice_dir, issue + ext)
            if os.path.isfile(p):
                return p
        return None

    @classmethod
    def available(cls, voice_dir=None):
        """扫描可用的语音名（mp3 与 wav 合并）。

        排除：音量缓存子目录、归档子目录，以及 `_` 开头的内部/临时文件
        （调试时留下的 _试听.mp3 曾混进试听列表，别让它出现在家长界面上）。
        """
        d = voice_dir or pm.VOICE_DIR
        if not os.path.isdir(d):
            return []
        names = set()
        for f in os.listdir(d):
            if f.startswith("_"):
                continue
            p = os.path.join(d, f)
            low = f.lower()
            if not os.path.isfile(p):
                continue
            if low.endswith(".mp3") or low.endswith(".wav"):
                names.add(f.rsplit(".", 1)[0])
        return sorted(names)

    # ---------------- MCI（首选）

    def _mci(self, cmd):
        import ctypes
        winmm = ctypes.windll.winmm
        buf = ctypes.create_unicode_buffer(512)
        err = winmm.mciSendStringW(cmd, buf, 511, None)
        if err:
            eb = ctypes.create_unicode_buffer(512)
            winmm.mciGetErrorStringW(err, eb, 511)
            raise OSError(f"MCI err={err} {eb.value} cmd={cmd}")
        return buf.value

    def _say_mci(self, path):
        """MCI 播放。文件名含中文也 OK —— 走的是 Unicode 版 mciSendStringW。"""
        try:
            self._mci(f"close {self._ALIAS}")        # 上一段没播完就顶掉
        except OSError:
            pass
        kind = "mpegvideo" if path.lower().endswith(".mp3") else "waveaudio"
        self._mci(f'open "{path}" type {kind} alias {self._ALIAS}')
        try:
            self._mci(f"setaudio {self._ALIAS} volume to "
                      f"{int(self.GAIN.get(self.volume, 1.0) * 1000)}")
        except OSError:
            pass          # 某些设备不支持 setaudio，音量退回系统音量即可，不算失败
        self._mci(f"play {self._ALIAS}")

    # ---------------- winsound 兜底

    def _scaled(self, src):
        """按档位缩放 wav 振幅并缓存（winsound 没法调音量）。

        ⚠️ Python 3.13 已删掉 audioop，所以用 numpy 做。
        mp3 不走这里 —— mp3 用 MCI 的 setaudio 直接调。
        """
        gain = self.GAIN.get(self.volume, 1.0)
        if gain >= 1.0:
            return src
        try:
            import wave
            import numpy as np
        except ImportError:
            return src
        dst = os.path.join(self.voice_dir, f"_vol_{self.volume}", os.path.basename(src))
        try:
            if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                return dst
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with wave.open(src, "rb") as w:
                params = w.getparams()
                data = w.readframes(w.getnframes())
            if params.sampwidth != 2:
                return src      # 只处理 16bit PCM，其它格式原样播
            a = np.frombuffer(data, dtype=np.int16).astype(np.float32) * gain
            np.clip(a, -32768.0, 32767.0, out=a)
            with wave.open(dst, "wb") as w:
                w.setparams(params)
                w.writeframes(a.astype(np.int16).tobytes())
            return dst
        except Exception:
            return src

    def _say_winsound(self, path):
        if not path.lower().endswith(".wav"):
            raise OSError("winsound 不支持 mp3")
        import winsound
        winsound.PlaySound(self._scaled(path), winsound.SND_FILENAME | winsound.SND_ASYNC)

    # ---------------- 对外

    def say(self, issue):
        """把一段语音排进播放队列，**立即返回**。

        返回值只表示"已排入队列"，不代表播放成功 —— 结果由播放线程异步汇报。
        """
        if self.mute:
            return False
        path = self._find(issue)
        return self._offer(("file", path) if path else ("beep", None))

    def _beep(self):
        try:
            import winsound
            winsound.Beep(880, 180)
        except Exception:
            pass

    def chime(self, kind):
        """系统事件提示音。

        ⚠️ 目前**只有** `kind="done"`（标定完成）在用。
           历史上还有一个 `"good"`（坐正了的正反馈），已按需求移除 ——
           「坐姿没问题就不要发出任何声音」。所以 kind 改成必填，
           免得以后有人不传参又意外响出一声。
        ⚠️ `winsound.Beep` 是**同步阻塞**的（阻塞时长就是音长），
           所以必须走队列、不能直接在推理线程里调。
        """
        if self.mute:
            return
        self._offer(("chime", {"done": 880}.get(kind, 880)))



# ---------------------------------------------------------------- 审核存图

# 问题名 -> 触发它的指标。审核样本只能用来调**它自己触发的那个指标**：
# 家长标"确实歪了"可能是因为塌腰，而那一帧的 neck_angle 可能很低，
# 拿它去压 neck_angle 的阈值就是错的。
ISSUE_METRIC = {
    "头前倾": "neck_angle",
    "塌腰": "trunk_angle",
    "低头": "head_collapse",
}


class ReviewStore:
    """把「判定场景」存成图片 + 指标，供家长事后审核。

    为什么要它（而不是"录 15 分钟视频再人工标注"）：
      逐帧标注 15 分钟视频，家长不会做，数据收不上来 —— 而"误报率 < 10%"
      这个验收指标**只有人工标注能算出来**。改成"提醒响了自动存一张，
      家长顺手点一下对不对"，成本几乎为零，数据才真的收得上来。

    ⚠️ 能力边界（界面上必须让家长知道）：
      能发现【误报】—— 判"歪了"其实坐得好；
      **发现不了【漏报】**，因为漏报时程序压根没存图。
      所以除提醒现场外，还按固定间隔存一些"判为良好"的样本，
      否则漏报率永远测不出来。

    隐私：全部本地存储、不上传；家长可一键清空；超出上限自动删最旧的。
    """

    def __init__(self, root=None, max_files=300, enabled=True):
        self.root = root or os.path.join(HERE, "review")
        self.pending_dir = os.path.join(self.root, "pending")
        self.done_dir = os.path.join(self.root, "done")
        self.max_files = int(max_files or 300)
        self.enabled = bool(enabled)

    # ---------------- 基本操作

    def _ensure(self):
        os.makedirs(self.pending_dir, exist_ok=True)
        os.makedirs(self.done_dir, exist_ok=True)

    def save(self, frame, metrics, base, issue, kind="reminder", cfg=None, full_frame=None):
        """存一张现场图 + 指标。返回 id；失败返回 None（存图失败绝不影响提醒）。

        full_frame：可选的原图。只有 kind="reminder" 才传 —— 见下面 keep_full 的说明。
        """
        if not self.enabled or frame is None or pm.cv2 is None:
            return None
        try:
            self._ensure()
            t = time.time()
            ident = time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t*1000) % 1000:03d}"
            jpg = os.path.join(self.pending_dir, ident + ".jpg")
            ok, buf = pm.cv2.imencode(".jpg", frame,
                                      [int(pm.cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                return None
            with open(jpg, "wb") as f:
                f.write(buf.tobytes())

            # 另存一张原图（默认只给"提醒现场"存）。
            #
            # ⚠️ 为什么要单独存原图：审核用的 480px 缩略图**不能用来重训练模型**
            #    （分辨率偏小 + JPEG 压缩伤精度 + 没有关键点真值坐标）。
            #    但如果以后要走 P3（儿童专用关键点模型微调），那些帧是**回不来的** ——
            #    只能提前存。所以留一份原图，把"以后能不能做 P3"这个选项保留住。
            #    代价：原图约 150~250KB，比缩略图大 6~8 倍，所以**只给提醒现场存**，
            #    定期抽样不存（抽样一天几十张，存原图会撑爆硬盘）。
            keep_full = bool((cfg or {}).get("review_keep_fullres", True))
            if kind == "reminder" and keep_full and full_frame is not None:
                try:
                    okf, buff = pm.cv2.imencode(
                        ".jpg", full_frame, [int(pm.cv2.IMWRITE_JPEG_QUALITY), 92])
                    if okf:
                        with open(jpg[:-4] + "_full.jpg", "wb") as f:
                            f.write(buff.tobytes())
                except Exception:
                    pass          # 原图存不下不影响审核主流程

            def clean(d):
                return {k: (None if v is None else round(float(v), 4))
                        for k, v in (d or {}).items()}

            data = {
                "id": ident,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
                "kind": kind,                 # reminder=提醒现场 / sample=判为良好的抽样
                "issue": issue,               # 触发的问题名（sample 为 None）
                "should_trigger": None,       # 家长审核结果：None=未审
                "has_full": os.path.isfile(jpg[:-4] + "_full.jpg"),
                "metrics": clean(metrics),
                "baseline": clean(base),
                "camera": (cfg or {}).get("camera"),
                "yaw": (cfg or {}).get("side_yaw"),
                "note": ("原图仅用于将来可能的模型微调；不加关键点标注就不能用于训练"
                         if os.path.isfile(jpg[:-4] + "_full.jpg") else ""),
            }
            with open(jpg[:-4] + ".json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.prune()
            return ident
        except Exception:
            return None

    @staticmethod
    def _load(js):
        try:
            with open(js, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def items(self, which="pending"):
        """列出待审/已审条目，按时间倒序。返回 [{'data':..., 'jpg':..., 'json':...}]"""
        d = self.pending_dir if which == "pending" else self.done_dir
        if not os.path.isdir(d):
            return []
        out = []
        for f in sorted(os.listdir(d), reverse=True):
            if not f.endswith(".json"):
                continue
            js = os.path.join(d, f)
            jpg = js[:-5] + ".jpg"
            data = self._load(js)
            if data and os.path.isfile(jpg):
                out.append({"data": data, "jpg": jpg, "json": js})
        return out

    @staticmethod
    def _files_of(item):
        """一个条目涉及的所有文件（含可能存在的原图）。"""
        jpg, js = item["jpg"], item["json"]
        out = [jpg, js]
        full = jpg[:-4] + "_full.jpg"
        if os.path.isfile(full):
            out.append(full)
        return out

    def label(self, item, should_trigger):
        """记录家长的判断并归档。

        should_trigger 是**归一化**后的结论，而不是原始按钮名：
          提醒现场 → 「确实歪了」=True ／「其实坐得好」=False（误报）
          良好抽样 → 「确实坐得好」=False ／「其实歪了」=True（漏报）
        这样两类样本可以用同一套算法处理。
        """
        self._ensure()
        try:
            d = item["data"]
            d["should_trigger"] = bool(should_trigger)
            d["reviewed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(item["json"], "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            for src in self._files_of(item):
                dst = os.path.join(self.done_dir, os.path.basename(src))
                if os.path.exists(dst):
                    os.remove(dst)
                os.replace(src, dst)
            return True
        except Exception:
            return False

    def clear_all(self):
        """一键清空（隐私）。只删自己目录里的文件，不递归外扩。"""
        n = 0
        for d in (self.pending_dir, self.done_dir):
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                p = os.path.join(d, f)
                low = f.lower()
                if os.path.isfile(p) and (low.endswith(".jpg") or low.endswith(".json")):
                    try:
                        os.remove(p)
                        n += 1
                    except OSError:
                        pass
        return n

    def prune(self):
        """超出上限时删最旧的（先删未审核的，保证已审核的标注不丢）。"""
        try:
            pend = self.items("pending")
            done = self.items("done")
            total = len(pend) + len(done)
            if total <= self.max_files:
                return 0
            drop = total - self.max_files
            victims = pend[-drop:] if len(pend) >= drop else (pend + done)[-drop:]
            n = 0
            for it in victims:
                for p in self._files_of(it):
                    try:
                        os.remove(p)
                        n += 1
                    except OSError:
                        pass
            return n
        except Exception:
            return 0

    # ---------------- 统计与建议

    def stats(self):
        pend = self.items("pending")
        done = self.items("done")
        labeled = [i["data"] for i in done if i["data"].get("should_trigger") is not None]
        wrong = [d for d in labeled if not d["should_trigger"]]
        missed = [d for d in labeled if d["should_trigger"] and d.get("kind") == "sample"]
        fp_rate = (len([d for d in wrong if d.get("kind") == "reminder"]) /
                   max(1, len([d for d in labeled if d.get("kind") == "reminder"])))
        return {
            "pending": len(pend), "labeled": len(labeled),
            "wrong": len(wrong), "missed": len(missed),
            "fp_rate": fp_rate if labeled else None,
        }

    def suggest_all(self, cfg):
        """用已审核的**提醒现场**样本，给每个指标算建议阈值。

        只取 kind=reminder（有明确 issue，metric 可归因）；
        良好抽样只用来让家长发现漏报，不参与拟合（否则会污染另一侧的指标）。
        """
        labeled = [i["data"] for i in self.items("done")
                   if i["data"].get("should_trigger") is not None
                   and i["data"].get("kind") == "reminder"]
        out = {}
        for metric in ("neck_angle", "trunk_angle", "head_collapse"):
            should, shouldnt = [], []
            for d in labeled:
                if ISSUE_METRIC.get(d.get("issue")) != metric:
                    continue
                sc = pm.side_score(metric, d.get("metrics") or {}, d.get("baseline") or {})
                if sc is None:
                    continue
                (should if d["should_trigger"] else shouldnt).append(sc)
            cur = (cfg.get("side_thresholds") or {}).get(metric)
            res = pm.suggest_threshold(should, shouldnt, current=cur)
            if res:
                res["metric"] = metric
                out[metric] = res
        return out



# ---------------------------------------------------------------- 引擎

class PostureEngine(threading.Thread):
    def __init__(self, cfg=None, on_log=None):
        super().__init__(daemon=True, name="posture-engine")
        self.cfg = cfg or load_config()
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
        self._cap = None
        self._lm = None
        self._gate = None
        self._base = None

        self._snap = {
            "phase": "idle", "error": None, "preview": None, "kps": None,
            "level": "idle", "text": TEXT_IDLE, "issue": None, "pending": None,
            "calib": {"done": 0, "need": 15, "sec_left": 0.0},
            "baseline": None, "detect_rate": None, "metrics": None,
            "mute": bool(self.cfg.get("mute", False)),
            "volume": self.cfg.get("volume", "中"),
            "stats": self._blank_stats(),
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

    # ---------------- 线程主体

    def run(self):
        try:
            apply_config(self.cfg)
            model = pm.find_model(self.cfg.get("model", pm.DEFAULT_MODEL))
            self._set(phase="opening", text="正在打开摄像头…")
            self._log(f"模型：{model}")
            # ⚠️ GUI 路径不经过 posture_monitor 的 CLI 入口，那里的 _need_mp() 守卫
            #    在这里不会被执行。不补这一句，mediapipe 导入失败（例如本机 SAC 拦了
            #    matplotlib 的 _image.pyd）会一路走到 make_landmarker，报出一个与真病因
            #    毫无关系的 `AttributeError: 'NoneType' object has no attribute
            #    'PoseLandmarker'`。这里提前拦，让原始报错能进日志。
            if pm.vision is None or pm.mp_python is None:
                pm._need_mp()   # 报错文案里带原始异常原文（sys.exit 由下面兜住）
            # ⚠️ 时间戳基准必须**全生命周期只设一次**。
            # MediaPipe 的 VIDEO 模式要求 detect_for_video 的时间戳单调递增；
            # 若在标定里用 `now - t0` 而重标定时把 t0 归零，时间戳会倒退，推理结果直接错。
            self._t0 = time.perf_counter()
            self._cap, idx = pm.open_camera(self.cfg.get("camera"),
                                            None,
                                            self.cfg.get("width"),
                                            self.cfg.get("height"))
            self._log(f"摄像头就绪：索引 {idx}")
            self._lm = pm.make_landmarker(model)

            self._gate = pm.ReminderGate(self.cfg.get("sustain_sec"),
                                         self.cfg.get("clear_sec"),
                                         self.cfg.get("remind_gap_sec"),
                                         self.cfg.get("remind_gap_any_sec"))
            self._base = load_baseline(self.cfg)
            if self._base:
                self._log("已读回上次的基线（机位没变，无需重标）")
                self._set(baseline=self._base)
            else:
                if not self._calibrate():
                    return   # 标定失败已写入 error
            self._monitor()
        except SystemExit as e:
            # pm.open_camera / find_model 用 sys.exit 报错，在子线程里会静默退出线程
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
                self._set(phase="error", error=f"{type(e).__name__}: {e}")
                self._log("异常：\n" + traceback.format_exc())
        finally:
            if self._cap is not None:
                try:
                    self._cap.release()
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

    # ---------------- 标定

    def _calibrate(self):
        need_n = 15
        calib_sec = float(self.cfg.get("calib_sec", 10.0))
        self._set(phase="calibrating", level="idle", text="请坐正，正在学习你的坐姿…",
                  calib={"done": 0, "need": need_n, "sec_left": calib_sec})
        self._log(f"开始标定：端坐保持 {calib_sec:.0f} 秒")

        interval = 1.0 / max(1, int(self.cfg.get("fps_sample", pm.FPS_SAMPLE)))
        samples, last, kps_last = [], None, None
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
                fr, m, kps = pm.grab(self._cap, self._lm, now - self._t0,
                                     pm.compute_side_metrics)
                if fr is not None:
                    last = _shrink(fr)
                kps_last = kps
                if m:
                    samples.append(m)
                self._set(preview=last, kps=kps_last, metrics=m,
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
            self._set(phase="error",
                      error=f"标定失败：只采到 {len(samples)} 个有效样本。\n"
                            "请确认摄像头拍到了完整的上半身、光线充足，然后重试。")
            return False

        self._base = pm.build_baseline(samples)
        save_baseline(self.cfg, self._base)
        self._set(baseline=self._base,
                  calib={"done": len(samples), "need": need_n, "sec_left": 0.0})
        self._log("标定完成：" + ", ".join(f"{k}={v:.3f}" for k, v in sorted(self._base.items())))
        self._speaker.chime("done")
        return True

    # ---------------- 监测

    def _monitor(self):
        interval = 1.0 / max(1, int(self.cfg.get("fps_sample", pm.FPS_SAMPLE)))
        self._set(phase="monitoring", level="idle", text=TEXT_IDLE)
        self._log(f"进入监测（{1.0/interval:.0f} FPS 采样）")

        stats = self._blank_stats()
        t_prev = time.perf_counter()
        next_run = t_prev
        hit_n = tot_n = 0
        detect_report_at = t_prev + 20.0
        sample_sec = float(self.cfg.get("review_sample_sec", 180.0) or 0)
        next_sample_at = t_prev + sample_sec

        while not self._stop.is_set():
            now = time.perf_counter()
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

            fr, m, kps = pm.grab(self._cap, self._lm, now - self._t0,
                                 pm.compute_side_metrics)
            prev = _shrink(fr) if fr is not None else None

            if m is None:
                # ⚠️ 没检测到人**不参与统计**，否则孩子一离座就被算成"坐姿差"。
                tot_n += 1
                self._set(preview=prev, kps=None, metrics=None, level="nodetect",
                          text=TEXT_NODETECT, issue=None, pending=None, stats=stats)
            else:
                tot_n += 1
                hit_n += 1
                issues = pm.judge_side(m, self._base)
                st = self._gate.update(issues, now)

                stats["total_sec"] += dt
                if issues:
                    stats["streak_sec"] = 0.0
                else:
                    stats["good_sec"] += dt
                    stats["streak_sec"] += dt
                    stats["best_streak_sec"] = max(stats["best_streak_sec"], stats["streak_sec"])
                stats["good_ratio"] = (stats["good_sec"] / stats["total_sec"]
                                       if stats["total_sec"] > 1e-6 else None)

                if st.fire:
                    stats["reminders"] += 1
                    queued = self._speaker.say(st.fire)
                    self._log(f"提醒：{st.fire}"
                              f"（{'已排入播放队列' if queued else '静音中，未播放'}）")
                    # 存现场图供家长审核 —— 这是拿到"误报率"验收指标的唯一现实途径
                    if self.review.save(prev, m, self._base, st.fire,
                                        kind="reminder", cfg=self.cfg, full_frame=fr):
                        self._log(f"已存现场图待审核（{st.fire}）"
                                  "：家长设置 → 审核判定")

                # ⚠️ 坐正了**刻意不发声**（需求原话：「坐姿没问题就不要发出任何声音」）。
                #    早期这里会 chime("good") 给个正反馈提示音 —— 已移除。
                #    正向激励改用界面表达（绿色大字「坐得真棒！」+ 连续良好计时），
                #    声音只留给"需要纠正"这一种情况。

                if st.issue:
                    tip = ISSUE_TIP.get(st.issue, st.issue)
                    pending = (st.issue, st.elapsed) if not st.confirmed else None
                    self._set(preview=prev, kps=None, metrics=m, level="warn", text=tip,
                              issue=st.issue, pending=pending, stats=stats)
                else:
                    self._set(preview=prev, kps=None, metrics=m, level="good",
                              text=TEXT_GOOD, issue=None, pending=None, stats=stats)
                    # 定期存一张"判为良好"的抽样。
                    # ⚠️ 这不是多余的：漏报时程序不会存任何图，
                    #    没有这类抽样，漏报率就永远测不出来。
                    if sample_sec > 0 and now >= next_sample_at:
                        next_sample_at = now + sample_sec
                        if self.review.save(prev, m, self._base, None,
                                            kind="sample", cfg=self.cfg):
                            self._log("已存一张抽样现场图（判为良好，供家长查漏报）")

            if now >= detect_report_at:
                detect_report_at = now + 30.0
                rate = hit_n / max(1, tot_n)
                self._set(detect_rate=rate)
                if rate < 0.3 and tot_n > 30:
                    self._log(f"⚠️ 检出率仅 {rate*100:.0f}%——摄像头可能没拍到人，"
                              "或光线太暗/机位不对。检出率低时判定结果不可信。")

        self._log("监测已停止")


# ---------------------------------------------------------------- 运行日志落盘

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
            p = path or os.path.join(HERE, "logs", time.strftime("run-%Y%m%d.log"))
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


# ---------------------------------------------------------------- 界面预览图

# 预览图宽度上限。
# 界面是按 2x 缩放显示的（本机 devicePixelRatio = 2.00）。
# 实测预览控件约 520 逻辑像素宽 -> 需要 **1040 物理像素**才不虚；
# 取 1280 留出余量（窗口再宽一点也不用上采样）。
# 480 是早期"能看清有没有人"的口径，在 2x 屏上明显偏糊。
# 代价只是每帧多一次 resize（对 5 FPS 可忽略；4K 全管线实测仍只用 1/4 预算）。
# ⚠️ 与 posture_app.fit_pixmap() 配套：那边按物理像素缩放，
#    所以这里的预览图必须**不小于**控件的物理宽度，否则就是上采样发虚。
PREVIEW_MAX_W = 1280


def _shrink(frame, max_w=PREVIEW_MAX_W):
    """把帧缩成**界面显示用**的小图。

    ⚠️ 只用于预览。**判定与存图都不走它** —— 判定用原始帧，
    审核原图走 `ReviewStore(full_frame=...)`，各自保全最高清。
    """
    if frame is None or pm.cv2 is None:
        return None
    try:
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        k = max_w / float(w)
        return pm.cv2.resize(frame, (max_w, int(h * k)), interpolation=pm.cv2.INTER_AREA)
    except Exception:
        return None

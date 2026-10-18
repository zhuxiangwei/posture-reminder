# -*- coding: utf-8 -*-
"""语音播放（提醒音）。

从 `posture_engine` 搬出来单独成模块的原因：
家长设置界面（试听、音量、静音）、CLI（`--side-only` 实时窗口）、引擎线程
三处都要用它；留在引擎里就等于让前两者被迫 import 整个引擎。

⚠️ 本模块**不做间隔控制** —— "多久才允许再叫一次"属于迟滞逻辑，
   住在 `posture.gate.ReminderGate`。早期版本在播放函数里也判了一次 gap，
   导致同一规则散落在两处、改一处漏一处。
"""

import os
import queue
import threading

from posture import paths


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
        self.voice_dir = voice_dir or paths.VOICE_DIR
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
        d = voice_dir or paths.VOICE_DIR
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


# ---------------------------------------------------------------- CLI 用的同步播放

def play_reminder(issue, voice_dir=None):
    """命令行模式下播一条提醒：优先预生成的语音 wav，缺失则退回系统提示音。

    ⚠️ 这是 CLI 的**简单路径**（单线程、就叫一声，不建播放线程）。
       图形界面/引擎走 `Speaker`，那里有静音、音量档位与异步队列。
    """
    d = voice_dir or paths.VOICE_DIR
    wav = os.path.join(d, f"{issue}.wav")
    if os.path.isfile(wav):
        try:
            import winsound
            winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
            print(f"  >>> 语音提醒：{issue}")
            return True
        except Exception:
            pass
    try:
        import winsound
        winsound.Beep(880, 180)
    except Exception:
        print("\a", end="")
    print(f"  >>> 提醒：{issue}")
    return False

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线生成语音提醒片段
====================

把提醒语**预生成**成音频文件放进 voice/ 目录。运行时只做本地播放——
零延迟、零网络依赖、语气可控（报告 §8 的主力方案）。

    python tools/make_voice.py                # 用 edge-tts 生成纯侧面模式需要的 3 条（mp3）
    python tools/make_voice.py --all          # 生成全部 7 条
    python tools/make_voice.py --list         # 看看有哪些语音可选
    python tools/make_voice.py --rate -15%    # 再慢一点
    python tools/make_voice.py --engine sapi  # 离线兜底（win32com + SAPI，出 wav）

两套引擎
--------
| 引擎 | 声音 | 生成时要网络 | 输出 | 评价 |
|---|---|---|---|---|
| **edge-tts**（默认） | 微软神经语音，8 个中文 | 要（播放时不要） | **wav** | ⭐ 年轻自然 |
| sapi | 本机 SAPI5，通常只有 Huihui | 不要 | wav | 兜底，成年女声、机械 |

**为什么默认 edge-tts**：本机 SAPI 只有 `Microsoft Huihui Desktop` 一个中文语音，
是成年女声，没有更年轻的选项。edge-tts 的 `zh-CN-XiaoyiNeural`（场景=Cartoon、
性格=Lively）是中文列表里最年轻自然的一个；`zh-CN-YunxiaNeural`（Cute）是备选。

**为什么最终存 wav（而不是 edge-tts 原生出的 mp3）**
--------------------------------------------------
| | wav（winsound 播） | mp3（只能 MCI 播） |
|---|---|---|
| 播放调用 | `PlaySound(SND_ASYNC)`，**真异步、无设备开关开销** | `mciSendString` 的 open/close 是**阻塞**调用 |
| 并发 | 天然安全 | 别名全局唯一，新播会 `close` 掉**正在播**的那条 |
| 体积 | ~100KB/条 | ~13KB/条 |

体积代价完全可接受（7 条不到 1MB），换来的是播放稳。mp3 那条路的坑是实测踩出来的。
（所以 `Speaker` 里 MCI 只作为 mp3 的兜底通道保留。）

⚠️ **本机没有 ffmpeg**，所以 mp3→wav 用 `soundfile`（自带 libsndfile 1.2+，
支持 mp3 解码），不依赖外部可执行文件。

⚠️ 另一条硬约束：**语速与时长此消彼长**。语速放慢会让片段变长，而报告 §7 要求 1~2 秒
（孩子正在思考，长句会打断）。所以"慢"要用**压短文案**来换，不能只调语速。
实测 Xiaoyi 在 -10% 下约 0.4 秒/字 —— 文案控制在 **4~6 字**。

依赖：
    edge-tts    （pip install edge-tts）
    soundfile   （pip install soundfile；仅 --engine edge 需要，用于 mp3→wav）
    pywin32     （仅 --engine sapi 需要，pip install pywin32）
"""
import argparse
import asyncio
import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.normpath(os.path.join(HERE, "..", "repos", "p0-prototype", "voice"))

EDGE_VOICE = "zh-CN-XiaoyiNeural"     # Cartoon + Lively，中文语音里最年轻的一个
EDGE_RATE = "-10%"                    # 放慢一点，语气更温和
SAPI_VOICE_KEY = "Huihui"
SAPI_RATE = 0

# 文件名必须与判定输出的问题名**逐字一致**（见 posture_monitor 的 judge_side / fuse）。
# ⚠️ 文案长度是硬约束：见文件头说明，4~6 字。
SIDE_TEXTS = [
    ("头前倾", "抬头收下巴"),
    ("塌腰", "背挺直一点"),
    ("低头", "头抬高一点"),
]
DUAL_TEXTS = [
    ("歪头", "头摆正一点"),
    ("歪肩", "肩膀放平"),
    ("太近", "往后靠一点"),
    ("含胸驼背", "挺起胸膛"),
]


def _mci(cmd):
    winmm = ctypes.windll.winmm
    buf = ctypes.create_unicode_buffer(512)
    err = winmm.mciSendStringW(cmd, buf, 511, None)
    if err:
        eb = ctypes.create_unicode_buffer(512)
        winmm.mciGetErrorStringW(err, eb, 511)
        raise OSError(f"MCI err={err} {eb.value} cmd={cmd}")
    return buf.value


def audio_ms(path):
    """用 MCI 读音频时长（毫秒）。读不到返回 None —— mp3/wav 都支持。"""
    alias = "dur_probe"
    try:
        try:
            _mci(f"close {alias}")
        except OSError:
            pass
        kind = "mpegvideo" if path.lower().endswith(".mp3") else "waveaudio"
        _mci(f'open "{path}" type {kind} alias {alias}')
        n = _mci(f"status {alias} length")
        _mci(f"close {alias}")
        return int(n)
    except Exception:
        return None


# ---------------------------------------------------------------- edge-tts

def list_edge():
    try:
        import edge_tts
    except ImportError:
        print("未安装 edge-tts： pip install edge-tts")
        return False

    async def run():
        return await edge_tts.list_voices()

    try:
        vs = asyncio.run(run())
    except Exception as e:
        print(f"取语音列表失败（需要联网）：{e}")
        return False

    cn = [v for v in vs if v["Locale"].startswith("zh-CN")]
    print(f"edge-tts 的 zh-CN 语音（共 {len(cn)} 个）：")
    for v in cn:
        tags = v.get("VoiceTag") or {}
        scene = ",".join(tags.get("ContentCategories") or [])
        mood = ",".join(tags.get("VoicePersonalities") or [])
        star = "   ⭐ 推荐（卡通+活泼，最年轻）" if v["ShortName"] == EDGE_VOICE else ""
        print(f"  {v['ShortName']:32s} {v['Gender']:6s} "
              f"场景={scene:18s} 性格={mood}{star}")
    return True


def gen_edge(text, out_path, voice, rate, volume):
    import edge_tts

    async def run():
        c = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
        await c.save(out_path)

    asyncio.run(run())


def mp3_to_wav(mp3_path, wav_path):
    """把 edge-tts 出的 mp3 转成 wav。

    ⚠️ 必须转：**winsound 不支持 mp3**，播 mp3 只能走 MCI，
       而 MCI 的 open/close 是阻塞调用、新播还会 `close` 掉正在播的那条（实测踩过）。
       wav 走 winsound 的 SND_ASYNC 才是真异步、无设备开关开销。
    ⚠️ 本机没有 ffmpeg，所以用 soundfile（自带 libsndfile 1.2+，支持 mp3 解码）。
    """
    import wave

    import soundfile as sf
    data, sr = sf.read(mp3_path, dtype="int16", always_2d=False)
    ch = 1 if data.ndim == 1 else data.shape[1]
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    return sr, ch


# ---------------------------------------------------------------- SAPI5

def list_sapi():
    try:
        import win32com.client
    except ImportError:
        print("未安装 pywin32： pip install pywin32")
        return False
    sp = win32com.client.Dispatch("SAPI.SpVoice")
    vs = sp.GetVoices()
    print("本机 SAPI5 语音：")
    for i in range(vs.Count):
        print(f"  - {vs.Item(i).GetDescription()}")
    print("  中文一般只有 Microsoft Huihui Desktop（成年女声，没有更年轻的）")
    return True


def gen_sapi(text, out_path, voice_key, rate, volume):
    import win32com.client
    sp = win32com.client.Dispatch("SAPI.SpVoice")
    if voice_key:
        vs = sp.GetVoices()
        for i in range(vs.Count):
            if voice_key.lower() in vs.Item(i).GetDescription().lower():
                sp.Voice = vs.Item(i)
                break
    sp.Rate = rate
    sp.Volume = volume
    # ⚠️ SAPI 默认输出到扬声器 —— 不换 AudioOutputStream 会当场出声
    fs = win32com.client.Dispatch("SAPI.SpFileStream")   # 每次都要新建实例：
    fs.Open(out_path, 3, False)                          # 对 Dispatch 实例加括号不是构造
    sp.AudioOutputStream = fs                            # SSFMCreateForWrite = 3
    sp.Speak(text)                                       # 同步阻塞，返回即写完
    fs.Close()


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(description="生成坐姿提醒语音片段")
    ap.add_argument("--all", action="store_true",
                    help="生成全部 7 条（默认只生成纯侧面用的 3 条）")
    ap.add_argument("--list", action="store_true", help="列出可选语音")
    ap.add_argument("--engine", choices=["edge", "sapi"], default="edge",
                    help="edge=神经语音(推荐) / sapi=本机离线兜底")
    ap.add_argument("--keep-mp3", action="store_true",
                    help="edge 引擎转换后，额外保留中间 mp3（默认删掉，避免与 wav 重复）")
    ap.add_argument("--voice", default=None,
                    help=f"语音名；edge 默认 {EDGE_VOICE}，sapi 默认 {SAPI_VOICE_KEY}")
    ap.add_argument("--rate", default=None,
                    help="语速；edge 用百分比（如 -10%%），sapi 用 -10~10")
    ap.add_argument("--volume", default=None,
                    help="音量；edge 用百分比（如 +0%%），sapi 用 0~100")
    ap.add_argument("--warn-sec", type=float, default=2.5,
                    help="超过这个时长就提示文案偏长（默认 2.5 秒）")
    args = ap.parse_args()

    if args.list:
        ok = list_edge()
        print()
        list_sapi()
        return 0 if ok else 1

    os.makedirs(VOICE_DIR, exist_ok=True)
    items = SIDE_TEXTS + (DUAL_TEXTS if args.all else [])
    ext = ".wav"          # 两种引擎最终都出 wav（理由见文件头）

    if args.engine == "edge":
        for mod, why in (("edge_tts", "pip install edge-tts"),
                         ("soundfile", "pip install soundfile（用于 mp3→wav）")):
            try:
                __import__(mod)
            except ImportError:
                sys.exit(f"未安装 {mod}： {why}\n（或改用离线兜底： --engine sapi）")
        voice = args.voice or EDGE_VOICE
        rate = args.rate or EDGE_RATE
        volume = args.volume or "+0%"
        print(f"引擎：edge-tts   语音：{voice}   语速：{rate}   音量：{volume}")
        print("输出：先出 mp3，再用 soundfile 转成 wav（本机没有 ffmpeg）")
    else:
        voice = args.voice or SAPI_VOICE_KEY
        rate = int(args.rate) if args.rate is not None else SAPI_RATE
        volume = int(args.volume) if args.volume is not None else 90
        print(f"引擎：SAPI5     语音：{voice}   语速：{rate}   音量：{volume}")

    print(f"输出目录：{VOICE_DIR}\n")
    made, long_ones = 0, []
    for name, text in items:
        path = os.path.join(VOICE_DIR, name + ext)
        tmp_mp3 = os.path.join(VOICE_DIR, name + "._tmp.mp3")
        try:
            if args.engine == "edge":
                gen_edge(text, tmp_mp3, voice, rate, volume)
                sr, ch = mp3_to_wav(tmp_mp3, path)
                if args.keep_mp3:
                    os.replace(tmp_mp3, os.path.join(VOICE_DIR, name + ".mp3"))
                else:
                    os.remove(tmp_mp3)
            else:
                gen_sapi(text, path, voice, rate, volume)
        except Exception as e:
            print(f"  ❌ {name}{ext}  生成失败：{type(e).__name__}: {e}")
            for p in (tmp_mp3,):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            continue

        size = os.path.getsize(path) if os.path.isfile(path) else 0
        if size <= 1000:
            print(f"  ❌ {name}{ext}  文件异常（{size} 字节）")
            continue

        ms = audio_ms(path)
        sec = ms / 1000.0 if ms else None
        flag = ""
        if sec is not None and sec > args.warn_sec:
            flag = f"   ⚠️ 偏长（{sec:.1f}s）—— 把文案再压短"
            long_ones.append(f"{name}({sec:.1f}s)")
        dur = f"≈{sec:.1f}s " if sec is not None else ""
        print(f"  ✅ {name}{ext}  {size/1024:5.1f} KB  {dur}「{text}」{flag}")
        made += 1

    print(f"\n成功 {made}/{len(items)} 条。")
    if long_ones:
        print(f"⚠️ 偏长的条目：{', '.join(long_ones)} —— "
              "报告 §7 要求 1~2 秒，长句会打断孩子思考。")
    print("试听：打开程序 →「家长设置 → 提醒与声音」→ 点对应的语音按钮，"
          "或直接双击 voice 目录里的文件。")
    print("确认不刺耳、语气不冲，再拿去给孩子听。")
    return 0 if made == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())

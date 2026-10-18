# -*- coding: utf-8 -*-
"""摄像头子系统：开流、档位阶梯、后端探测、profile 记忆、采集线程、掉线重连。

================================================================ 为什么要有这个文件

重构前这套逻辑散在 `posture_monitor.py` 里约 500 行，症状是用户报的两条：
**"加载很久"** 和 **"出错时根本不知道发生了什么"**。实测定位到的病因：

    ① 一次开流本身就要 5~7s（`cap.set()` 协商分辨率/格式要 2.6~4.5s，
       加上坏开流的每帧 1000ms 全零占位帧）。
    ② 更致命的是**重试矩阵组织得不对**：首选后端明明能用，
       却夹在 MSMF / ANY 之间反复付子进程探测的代价 —— 实测两次开流里
       有 **16s 纯粹浪费在探测已知会阻塞的后端上**（31s / 48.9s 的离谱数字全来自这里）。
    ③ 探测子进程**没设 MJPG**：它测的是"YUY2 能否出图"，
       而本项目全程用 MJPG —— 高分辨率档会被误判成"该后端不可用"。
    ④ 探测结果只存在内存里，**每次启动重新交一遍学费**。
    ⑤ 摄像头中途被别的程序抢走 / USB 掉线 -> **没有任何恢复路径**，
       画面永久停在最后一帧，程序看起来"还在跑"。

本文件的对策（与用户定的两条口径严格一致）：
    · **4K 优先、不降级**：档位阶梯永远从请求的最高档（默认 3840x2160）开始，
      同后端快速重试；只有确认拿不到画面才往下走，
      且**任何低于请求档的结果都作为 `degraded` 显式上报**，界面必须显示警告，
      绝不安静默地换成 480p。
    · **记住上一次成功的组合**（`camera_profile.json`）：下次启动直接照它开，
      先验后试（fast path），实测能把常见路径压到 3s 级。
    · **全局时间预算**：开流有 deadline，不会再出现"转圈 48 秒"。
    · **采集线程 `CameraStream`**：创建设备与读取帧都在**同一个长命线程**里
      （DSHOW 是 COM，见下面的 2x2 实测），界面与推理线程只从帧槽取最新帧；
      带心跳检测与**自动重连**。

================================================================ 实测硬约束（改代码前必读）

★ **创建必须在真正要用它的那个线程里**（2026-09-28 的 2x2 对照，每格 5 轮独立进程，
  1280x720 DSHOW）：

      创建线程    读取线程    成功率
      主线程      主线程      4/5 = 80%
      主线程      工作线程    3/5 = 60%
      工作线程    主线程      **0/5 = 0%**
      工作线程    工作线程    **0/5 = 0%**      ← 那个"工作线程"很快就结束了

  DSHOW 走 COM，cap 绑定**创建它的线程的 COM 单元**；该线程一结束，句柄就是死的。
  → 所以绝不允许"在工作线程里创建 cap 再交给别人"。
  → `CameraStream` 之所以安全，是因为它**自己创建、自己读、线程长命不退出**。
  → 跨线程**读**是允许的（60~80% 那一档），只是略降可靠性；
    所以只在"探测阶段"用超时读兜阻塞，**稳态采集循环用不带超时的 read**。

★ **`cap.read()` 会在 C 层阻塞几分钟**（实测 MSMF grab 报 E_PENDING 之后）。
  try/except 抓不住，只有线程 join 超时或子进程硬杀能救。

★ **能打开 ≠ 有画面**：坏开流返回 ok=True、shape 正常，但**内容全零**、
  且每帧精确 1000ms。必须用 `frame_is_usable` 判，并**换新句柄重开**
  （重读同一句柄没用 —— 那一轮开流本身就是坏的）。

★ **好的开流前几帧也是全零**，档位越高越多：480p=0 帧 / 720p=1 帧 / 1080p=3 帧。
  所以预热容限必须按档位给（`warmup_budget`），写死会把好开流误杀或在坏开流上白等。

★ **MJPG 必须设在分辨率之前**：格式协商发生在选模式时。
  USB 2.0 实际带宽 ~35MB/s，YUY2 未压缩在 720p 就撞上极限（开流成功率 60%）；
  切 MJPG 后 4K 反而 10/10 稳定。驱动不提供该档 MJPG 时会**静默忽略**（1080p 就这样）。
"""

import json
import os
import subprocess
import sys
import threading
import time

from posture import deps, paths, settings

cv2 = deps.cv2
MJPG_FOURCC = deps.MJPG_FOURCC


# ---------------------------------------------------------------- 时间与档位预算

CAM_TIERS = [
    (3840, 2160),      # 4K —— 默认请求档，实测 MJPG 100% 可靠、~14 FPS
    (2560, 1440),      # 2K —— 同样 100% 可靠，但属于"降级"，须上报
    (1920, 1080),      # 1080p —— 驱动**不提供 MJPG**，YUY2 下不稳，属降级
    (1280, 720),
    (640, 480),        # 兜底档：一定能开，但画质差，必须**显式警告**
]
"""档位阶梯：从高到低。⚠️ 只作"往下退"的顺序，不作"默认从这里挑一个"。
   实际起点永远是用户请求的档位（默认 4K）—— 这是"4K 优先不降级"的实现方式。"""

MAX_OPEN_ATTEMPTS = 5
"""同一 (后端, 索引, 档位) 最多重开几次。

⚠️ 为什么需要重开而不是简单重读帧（2026-09-28 实测）：
   本机 USB 2.0 Camera 开流**有失败率**，但不是"档位不支持" —— 同档重开就能好。
   各档成功率（每档 8 轮独立开合）：480p 8/8；720p 4~5/8；1080p 5/8。
   重读同一句柄没用（那一轮开流本身就坏），必须**换新句柄重开**。
   取 5：按单次 50% 保守成功率算，3 次=87.5%、5 次=96.9%，再高收益递减。
   2026-10-18 改用构造期参数后，单次尝试从 ~7s 降到 ~2-6s，
   所以真正限制次数的其实是下面的 `OPEN_BUDGET_SEC`。"""

OPEN_BUDGET_SEC = 15.0
"""**整个开流过程**的时间预算（不是单次尝试）。

为什么还有 15s 而不是 5s：用户口径是"坚持 4K 优先，慢可以等"。
4K 的开流有实测 ~30-40% 的单次失败率（坏开流要读满 6 张全零帧才发现），
所以必须允许**同档重开 2~3 次**；预算太紧会造成"4K 明明能开却被判失败"。
但它必须存在 —— 重构前**没有**全局预算，才会出现 31s / 48.9s 那种失控数字。
预算用尽仍未拿到画面 -> 抛 `CameraError`（短文案 + 可展开详情），不再无限重试。

实测（本机 4K，2026-10-18）：单次尝试 0.7~5.1s；整个开流中位目标 ≤8s。
"""

OPEN_PROBE_TIMEOUT = 8.0
"""回退后端**子进程**探测的超时（秒）。

⚠️ 为什么用子进程而不是线程超时：某些后端（实测 MSMF）的
   `cv2.VideoCapture()` **构造函数本身无限阻塞**，那是 C 层阻塞，
   线程 join 超时救不回来（卡住的线程会一直攥着设备）。子进程可以硬杀。
"""

STALL_SEC = 6.0
"""采集中断多久算"掉线"（4K MJPG 实测 ~14 FPS，正常帧间隔 <0.1s，6s 极宽松）。"""

MIN_ATTEMPT_SEC = 2.5
"""**剩余预算**少于这个数就不再开始新一轮尝试（也不再起子进程探测）。

为什么需要这道闸：`deadline` 只在"每次读帧之前"起作用，而一轮尝试的代价是
构造 + 最多 6 次读帧 + 可能的二次校验，最坏 6~8s；子进程探测自己就要 3~8s
（要起一个 python 再把 cv2 导一遍）。实测本机出现过「预算 15s、22s 才报错」——
起点正是预算快用完时才想起去探 MSMF。这不是"预算没生效"，是**准入没闸**：
剩下的时间不够跑完一轮，就不该开始，否则报错永远比承诺晚好几秒。
"""

REOPEN_DELAY_SEC = 3.0
"""判定掉线后，多久允许发起下一次重开尝试（避免设备被抢时疯狂重试）。"""

# 后端顺序只作"优先猜测"，真正决定用哪个必须靠**能否读到有效帧**。
# 历史实测（留档，说明"不能写死"这条为什么重要）：
#   Nebula 02（VID_3AAE&PID_6373）：DSHOW 全废，只有 MSMF 能开，固定 1080p。
#   USB 2.0 Camera（VID_0C49&PID_636F）：结论**完全反转** —— DSHOW 能开；
#     MSMF 的 grab 报 E_PENDING，且**构造函数无限阻塞**。
#   2026-10-18 补充（构造期参数写法下）：MSMF 直接拒绝属性
#     （`can't set property 6` → isOpened()=False，0.00s 返回，反而更好排除）；
#     ANY 接受并给出 3840x2160 MJPG（0.94s）。
# **换设备必须重测**，所以顺序写死是错的，靠探测 + 记忆。
BACKEND_NAMES = ["DSHOW", "MSMF", "ANY"]


def _backend_table():
    """把后端名映射到 cv2 常量。cv2 缺失时返回空表（让上层报"缺依赖"）。"""
    if cv2 is None:
        return {}
    return {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}


# ---------------------------------------------------------------- 帧有效性

def read_frame(cap):
    """安全读帧。返回 (ok, frame)。

    ⚠️ 必须包 try —— 实测（MSMF 后端 + 设置分辨率）：
        cv2.error: (-215:Assertion failed) _step >= minstep in function 'cv::Mat::Mat'
        抛出点在 `cap.read()`。
        驱动异常时 read() **会直接抛 cv2.error，而不是规矩地返回 (False, None)**。
        原实现所有 read 都没保护，摄像头打一次嗝就能让整个程序崩掉。
    """
    try:
        ok, frame = cap.read()
    except Exception:
        return False, None
    if not ok or frame is None:
        return False, None
    return True, frame


def read_frame_timed(cap, timeout=2.5):
    """带**超时保护**的读帧，返回 (ok, frame, timed_out)。只在探测阶段用。

    ⚠️ 为什么普通 try/except 不够：实测 MSMF 后端 grab 失败后（E_PENDING），
       `cap.read()` 会**阻塞数分钟不返回**。那是 C 层阻塞，try/except 抓不住，
       只会让整个程序（含 GUI 线程）永久卡死。所以探测阶段用工作线程 + join 超时兜住。

    ⚠️ **稳态采集循环不要用本函数**：它把读操作挪到了另一个线程，
       而跨线程读会掉可靠性（见文件头的 2x2 实测：80% -> 60%）。
       稳态靠 `CameraStream` 的**心跳看门狗**发现停摆，而不是给每次读套超时。

    卡住的线程是 daemon，不阻止进程退出；但它持有的设备句柄要到 release 才释放，
    因此调用方仍需负责 release。
    """
    box = {}

    def _rd():
        try:
            ok, fr = cap.read()
            box["r"] = (ok, fr)
        except Exception:
            box["r"] = (False, None)

    th = threading.Thread(target=_rd, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return False, None, True        # 卡住，视为不可用
    ok, fr = box.get("r", (False, None))
    if not ok or fr is None:
        return False, None, False
    return True, fr, False


def frame_is_usable(frame, min_max_pixel=8):
    """判断一帧是否**真的有画面**（而不只是"读到了"）。

    ⚠️ 2026-09-28 踩出来的坑，务必保留：
       `cap.read()` 返回 ok=True **不代表画面有效**。
       实测本机 USB 2.0 Camera 在它不支持的档位下：
         · read() 返回 True
         · frame.shape 正常（如 1080x1920x3）
         · 但**内容全是 0**（frame.max() == 0），且每帧耗时精确 1000ms
       这是驱动"接受了格式协商但填不进数据"的占位行为。
       只判 ok 就会选中这个组合，然后画面全黑、判定逻辑对着黑图跑。

    判据用 max() 而非 mean()：只要有一个像素非零就说明有真实数据流，
    避免把"画面确实很暗"的正常情况误判成故障。
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    try:
        return int(frame.max()) >= min_max_pixel
    except Exception:
        return False


def warmup_budget(width=None, height=None):
    """该分辨率档位允许的"全零预热帧"上限（实测值 + 余量）。

    ⚠️ 为什么要按档位区分（2026-09-28 实测的逐帧序列）：
       好的开流前几帧也是全零的，档位越高越多：
         640x480   : 首帧就出图        -> 0 帧预热
         1280x720  : 第 1 帧出图       -> 1 帧预热
         1920x1080 : 第 3 帧才出图     -> 3 帧预热
       而**坏开流**的帧是每 1000ms 一张全零（掉帧占位）。
       所以这个上限既是"别误杀好开流"的下界，也是"别在坏开流上白等"的上界：
       一刀切设 4（为了兼容 1080p）会让 720p 的每次坏开流白等 4s 而不是 2s，
       实测把启动耗时从 ~4s 拖到 ~10s。故按档位给值。

    另：耗时不能用来提前判别 —— 好开流的首帧同样要等 1000ms。只能靠帧数。

    ★ 2026-10-18 换用"构造期参数"后重新实测 4K（本机 6 轮）：
      好开流的首个**有效**帧出现在第 **1~6** 张之间（3840x2160 MJPG 全部 6/6 成功），
      也就是说 4K 的预热全零帧最多能到 5 张。所以 4K 档必须给到 6，
      设成 4 会**误杀好开流**（表现为"明明能开却报画面全零，然后整轮重开"，
      白白多花 5~8s —— 这正是本次实测中位 10.6s 的成因之一）。
    """
    h = height or settings.DEFAULT_CAM_H
    if h > 2000:          # 4K（实测最多 5 张全零预热帧）
        return 6
    if h > 900:           # 1080p/1200p（实测 3 张）
        return 4
    return 2              # 720p 及以下（实测 0~1 张）


def wait_usable_frame(cap, tries=5, timeout=2.5, blank_abort=None, deadline=None):
    """连续读若干帧，只要有一帧**有效**就返回它。返回 (ok, frame, stats)。

    为什么要多试几帧：UVC 开流最初几帧常是空的（曝光/白平衡未就绪），
    只读一帧容易把好设备误判成坏设备。

    ⚠️ `blank_abort` 默认随档位走（`warmup_budget`），**不能一刀切**，理由见该函数文档。
    ⚠️ `deadline`（`time.perf_counter()` 时刻）用来把"开流很慢"变成"开流有上限"：
       到点立刻放弃本轮，不再白等 —— 这是重构前缺失的那道闸。

    stats 带诊断信息（tries / max_pixel / timed_out / sec），报错与计时分析都靠它。
    """
    if blank_abort is None:
        blank_abort = warmup_budget()
    t0 = time.perf_counter()
    last = None
    timed_out = False
    blank_run = 0
    for i in range(tries):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        ok, fr, to = read_frame_timed(cap, timeout)
        if to:
            timed_out = True
            break
        if ok:
            last = fr
            if frame_is_usable(fr):
                return True, fr, {"tries": i + 1, "max_pixel": int(fr.max()),
                                  "timed_out": False,
                                  "sec": time.perf_counter() - t0}
            blank_run += 1
            if blank_abort and blank_run >= blank_abort:
                break      # 预热帧数已超上限仍全零 -> 这轮开流是坏的
    mp = int(last.max()) if last is not None else -1
    return False, last, {"tries": tries, "max_pixel": mp, "timed_out": timed_out,
                         "sec": time.perf_counter() - t0}


# ---------------------------------------------------------------- 后端可用性探测

_PROBE_MEM_CACHE = {}
"""进程内缓存：同一次运行里别重复付子进程代价。"""


def _probe_code(name, index, width, height):
    """子进程里跑的探测脚本。

    ⚠️ 两个必须与主流程**保持一致**的点（不一致就等于探测结果没有意义）：
      1. **先 MJPG 后分辨率**。重构前这里漏了 MJPG，于是它测的是"YUY2 能否出图" ——
         在 4K/2K 档必然测出"该后端不可用"，把能用的后端误杀。
         （本项目全程 MJPG，探测条件与真实开流条件不一致 = 结论无意义。）
      2. **走构造期参数**（与 `open_once` 同一套协商方式）。
         用"open 之后再 set"的老写法，探测会把 4.5s 的 set 成本算进超时里，
         于是"慢但不坏"的后端被判不可用（实测本机 4K 三次 set 要 4.46s）。
    """
    params = (f"[cv2.CAP_PROP_FOURCC, mjpg, cv2.CAP_PROP_FRAME_WIDTH, {width}, "
              f"cv2.CAP_PROP_FRAME_HEIGHT, {height}]")
    return (
        "import cv2, sys\n"
        "mjpg = cv2.VideoWriter_fourcc(*'MJPG')\n"
        "cap = None\n"
        "try:\n"
        f"    cap = cv2.VideoCapture({index}, cv2.CAP_{name}, {params})\n"
        "except TypeError:\n"
        f"    cap = cv2.VideoCapture({index}, cv2.CAP_{name})\n"
        "if cap is None or not cap.isOpened():\n"
        "    print('NO'); sys.exit(0)\n"
        "if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) < 1:\n"
        "    cap.set(cv2.CAP_PROP_FOURCC, mjpg)\n"
        f"    cap.set(cv2.CAP_PROP_FRAME_WIDTH, {width})\n"
        f"    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, {height})\n"
        "best = 0\n"
        "for _ in range(10):\n"
        "    ok, fr = cap.read()\n"
        "    if ok and fr is not None and fr.size > 0:\n"
        "        best = max(best, int(fr.max()))\n"
        "        if best > 8:\n"
        "            break\n"
        "print('YES' if best > 8 else 'NO')\n"
    )


def backend_viable_subprocess(name, index, width, height, timeout=None, profile=None):
    """在**子进程**里验证 (后端, 索引, 档位) 能否读到有效画面。返回 True/False。

    超时（构造函数或读帧阻塞）一律视为不可用，并把子进程硬杀 ——
    这样会卡死的后端不会拖住主进程。

    ⚠️ 结果**落盘**（`camera_profile.json` 的 `probes` 字段）。
       重构前只缓存在内存里，每次启动都要重新付 8s×N 的探测学费，
       这正是"加载很久"的主要构成之一。
    """
    timeout = timeout if timeout is not None else OPEN_PROBE_TIMEOUT
    key = f"{name}|{index}|{width}|{height}"
    if key in _PROBE_MEM_CACHE:
        return _PROBE_MEM_CACHE[key]

    prof = profile if profile is not None else CameraProfile.load()
    cached = prof.probe(key)
    if cached is not None:
        _PROBE_MEM_CACHE[key] = cached
        return cached

    ok = False
    try:
        r = subprocess.run([sys.executable, "-c", _probe_code(name, index, width, height)],
                           capture_output=True, text=True, timeout=timeout)
        ok = "YES" in (r.stdout or "")
    except subprocess.TimeoutExpired:
        ok = False          # 卡住了 -> 该后端/档位不可用（子进程已被 kill）
    except Exception:
        ok = False
    _PROBE_MEM_CACHE[key] = ok
    prof.set_probe(key, ok)
    return ok


# ---------------------------------------------------------------- profile：记住上次成功的组合

class CameraProfile:
    """`camera_profile.json` 的读写：上次成功的组合 + 探测结果缓存。

    为什么"先验后试"是安全的（而不是赌上一次的结果）：
        profile 只决定**先试哪一个**，不跳过验证。
        命中后仍然要 `wait_usable_frame` 真的读到有效帧才算数，
        只是把探测预算压到 `FAST_PROBE_BUDGET_SEC` —— 超时/失败就立刻回到完整阶梯。
        换摄像头、拔插、改分辨率都不会被 profile 卡住。

    ⚠️ 档位有效性**与 (后端, 索引, 宽高) 三者相关**，所以缓存 key 带齐这三项。
    """

    VERSION = 1

    def __init__(self, data=None, path=paths.CAMERA_PROFILE_PATH):
        self.path = path
        self.d = dict(data or {})
        self.d.setdefault("version", self.VERSION)
        self.d.setdefault("last_good", None)
        self.d.setdefault("probes", {})
        self.d.setdefault("tiers", {})

    # ---- 磁盘

    @classmethod
    def load(cls, path=paths.CAMERA_PROFILE_PATH):
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return cls(json.load(f), path)
        except Exception:
            pass       # profile 坏了就当没有，绝不能让程序起不来
        return cls(None, path)

    def save(self):
        try:
            self.d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.d, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)      # 原子替换：避免断电留下半个 JSON
            return True
        except Exception:
            return False

    # ---- last_good

    @property
    def last_good(self):
        return self.d.get("last_good")

    def set_last_good(self, backend, index, width, height, fourcc, opened_sec):
        self.d["last_good"] = {
            "backend": backend, "camera": int(index),
            "width": int(width), "height": int(height),
            "fourcc": fourcc, "opened_sec": round(float(opened_sec), 2),
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save()

    def matches_last_good(self, backend, index, width, height):
        g = self.last_good or {}
        return (g.get("backend") == backend and g.get("camera") == int(index)
                and g.get("width") == int(width) and g.get("height") == int(height))

    # ---- 探测结果

    def probe(self, key):
        v = (self.d.get("probes") or {}).get(key)
        return None if v is None else bool(v)

    def set_probe(self, key, ok):
        probes = self.d.setdefault("probes", {})
        probes[key] = bool(ok)
        self.save()

    # ---- 档位实测结论（给 --list-cameras 与报错文案用）

    def tier(self, backend, index, width, height):
        return (self.d.get("tiers") or {}).get(f"{backend}|{index}|{width}|{height}")

    def set_tier(self, backend, index, width, height, usable, actual=None, note=None):
        self.d.setdefault("tiers", {})[f"{backend}|{index}|{width}|{height}"] = {
            "usable": bool(usable),
            "actual": list(actual) if actual else None,
            "note": note,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save()

    def forget(self):
        """家长点"重新探测摄像头"时调用：清掉所有记忆，回到冷启动阶梯。"""
        self.d.update({"last_good": None, "probes": {}, "tiers": {}})
        return self.save()


# ---------------------------------------------------------------- 开流失败的结构化异常

class CameraError(Exception):
    """摄像头错误：**短文案 + 详情**两段。

    为什么要两段：以前失败时只有一大坨排查清单，界面弹窗放不下、
    用户也看不懂自己该干什么。现在 `short` 上界面（一行），
    `detail` 收进"详情"折叠区与日志（含每个组合的失败类型、耗时、档位）。
    """

    def __init__(self, short, detail=""):
        super().__init__(short)
        self.short = short
        self.detail = detail

    def __str__(self):
        return self.short if not self.detail else f"{self.short}\n\n{self.detail}"


CHECKLIST = (
    "排查顺序：\n"
    "  1) 是否被别的程序占用（相机 App / 会议软件 / 浏览器标签页）\n"
    "  2) 隐私设置是否允许桌面应用访相机\n"
    "     Windows 设置 > 隐私和安全性 > 相机 > 允许桌面应用访问\n"
    "  3) USB 线接触不良 / 供电不足（换口试，优先主板直连口）\n"
    "  4) 分辨率档位超出设备能力 —— 跑 --list-cameras 看哪些档位真有画面\n"
    "  5) 换后端试：--backend DSHOW / MSMF / ANY"
)


def _fmt_attempts(rows):
    """把尝试记录表排成人话（每行：后端/索引/档位/结果/耗时）。"""
    out = []
    for r in rows:
        rs = f"{r.get('width')}x{r.get('height')}" if r.get("width") else "默认"
        secs = r.get("sec")
        tail = f"  {secs:.1f}s" if isinstance(secs, (int, float)) else ""
        out.append(f"    后端={r.get('backend')} 索引={r.get('camera')} {rs:>9} "
                   f"-> {r.get('result')}{tail}")
    return "\n".join(out[:16])


# ---------------------------------------------------------------- 单次开流尝试

def _apply_sets(cap, res):
    """显式设置 MJPG + 分辨率（+ 缓冲 1 帧）。

    ⚠️ FOURCC **必须**先于分辨率：格式协商发生在选模式时，顺序反了会静默失效，
       4K 直接开不出来（这是本项目最容易复发的坑，回归用例 I2c 专门锁它）。
       驱动不支持该档的 MJPG 时会静默忽略（1080p 就这样），属正常。

    ⚠️ 本函数只作**兜底路径**。真机实测：开流之后再 `cap.set()` 这三项，
       每次要 **4.46s**（2026-10-18 本机 4K，6 轮全部 ~4.5s）；
       而把同样的参数交给构造函数一次性协商，只要 **0.17s**。
       所以 `open_once` 首选构造期参数，只有后端不认那套参数时才走这里。
    """
    cap.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    try:
        # 队列深度 1：DSHOW 默认缓冲好几帧，掉线重连后会把旧帧一起吐出来，
        # 表现为"画面是 2 秒前的"。不支持时静默失败，无害。
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _open_params(res):
    """构造期参数列表（OpenCV ≥4.7 / cv2 5.0 支持）。返回 None 表示不支持。"""
    if not res or not res[0]:
        return None
    try:
        return [cv2.CAP_PROP_FOURCC, MJPG_FOURCC,
                cv2.CAP_PROP_FRAME_WIDTH, int(res[0]),
                cv2.CAP_PROP_FRAME_HEIGHT, int(res[1]),
                cv2.CAP_PROP_BUFFERSIZE, 1]
    except Exception:
        return None


def open_once(index, backend_name, res, blank_abort=None, tries=6, timeout=2.5,
              deadline=None, verify=True):
    """在**调用线程**里开一次流。返回 (cap, info) 或 (None, info)。

    `res` = 请求档位 (w, h)；`(None, None)` 表示"不显式设分辨率"（兜底轮）。

    ⚠️ 四条硬约束的落点（改这里前先读文件头）：
      · `VideoCapture(...)` 必须发生在**调用线程**，不许包进工作线程
        （DSHOW 是 COM，句柄绑创建线程 —— 实测跨线程创建 0/5 全废）。
      · MJPG 先于分辨率（走 `res` 参数列表时由构造期一次性协商，顺序天然正确）。
      · 判"有效帧"而不是判 ok（坏开流返回 ok=True 但全零）。
      · **计时分段记录**：create / set / read 各花多少要能看见，
        否则"加载很久"永远只能靠猜。
    """
    table = _backend_table()
    be = table.get(backend_name)
    if be is None:
        raise CameraError(f"没有名为 {backend_name} 的采集后端",
                          f"可用后端：{'/'.join(table) or '（缺 cv2）'}")
    t0 = time.perf_counter()
    params = _open_params(res)
    used_params = False
    cap = None
    if params is not None:
        # ★ 首选：把格式/分辨率交给构造函数一起协商（实测比 set 快 26 倍）
        try:
            cap = cv2.VideoCapture(index, be, params)
            used_params = bool(cap is not None)
        except TypeError:
            cap = None                    # 老版本 OpenCV 没有这个重载
    if cap is None:
        cap = cv2.VideoCapture(index, be)
    t_create = time.perf_counter() - t0

    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None, {"backend": backend_name, "camera": index,
                      "width": res[0], "height": res[1],
                      "result": "设备打不开（不存在或被占用）",
                      "sec": time.perf_counter() - t0,
                      "sec_create": t_create, "sec_set": 0.0, "sec_read": 0.0}

    t1 = time.perf_counter()
    if res[0] and not used_params:
        _apply_sets(cap, res)             # 兜底：后端不认构造期参数
    elif res[0] and used_params:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    t_set = time.perf_counter() - t1

    frame = None
    stat = {"tries": 0, "max_pixel": -1, "timed_out": False, "sec": 0.0}
    t2 = time.perf_counter()
    if verify:
        if blank_abort is None:
            blank_abort = warmup_budget(*(res or (None, None)))
        ok, frame, stat = wait_usable_frame(cap, tries=tries, timeout=timeout,
                                            blank_abort=blank_abort, deadline=deadline)
        # ---- 构造期参数被该后端**忽略**了？（实测 MSMF 会拒绝属性设置）
        #      表现：能读到帧但实际尺寸不等于请求尺寸。补一次显式 set 再验一轮。
        if ok and used_params and res[0] and frame is not None:
            fh, fw = frame.shape[:2]
            if (fw, fh) != (res[0], res[1]):
                _apply_sets(cap, res)
                t_set += time.perf_counter() - t2
                t2 = time.perf_counter()
                ok, frame, stat = wait_usable_frame(cap, tries=tries, timeout=timeout,
                                                    blank_abort=blank_abort,
                                                    deadline=deadline)
        if not ok:
            try:
                cap.release()
            except Exception:
                pass
            result = ("后端阻塞/会卡死（已放弃）" if stat["timed_out"] else
                      ("画面全零（流开了但拿不到数据）" if stat["max_pixel"] == 0 else
                       (f"画面异常（最大像素 {stat['max_pixel']}）" if stat["max_pixel"] > 0
                        else "读不到帧")))
            return None, {"backend": backend_name, "camera": index,
                          "width": res[0], "height": res[1], "result": result,
                          "sec": time.perf_counter() - t0, "stat": stat,
                          "sec_create": t_create, "sec_set": t_set,
                          "sec_read": time.perf_counter() - t2}

    t_read = time.perf_counter() - t2
    # ---- 开流成功：读回实际档位与协商到的格式，作为"是否真拿到请求档"的依据
    try:
        h, w = frame.shape[:2] if frame is not None else (0, 0)
    except Exception:
        w = h = 0
    if not w:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = _fourcc_str(cap)
    return cap, {"backend": backend_name, "camera": index,
                 "width": res[0], "height": res[1],
                 "actual_w": w, "actual_h": h, "fourcc": fourcc,
                 "negotiated_at_open": used_params,
                 "result": "ok", "sec": time.perf_counter() - t0, "stat": stat,
                 "sec_create": t_create, "sec_set": t_set, "sec_read": t_read}


def _fourcc_str(cap):
    """读回设备**实际协商到**的四字符格式（判断 MJPG 到底有没有生效）。

    ⚠️ 只认**可打印 ASCII**：真机出现过设备把一个内部数值当 fourcc 报回来，
       原样拼字符串会得到 `=}ë6ä` 这种乱码，印在家长设置和日志里像是程序坏了。
       认不出来就说"未知(原始值)"—— 宁可信息少一点，不许输出乱码。
    """
    try:
        v = int(cap.get(cv2.CAP_PROP_FOURCC))
        raw = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    except Exception:
        return "?"
    if len(raw) == 4 and all(0x20 <= ord(c) < 0x7F for c in raw):
        return raw
    return f"未知(0x{v:08X})" if v else "?"


def _mjpg_ok(fourcc):
    return (fourcc or "").upper() in ("MJPG", "MJP1", "MPEG")


# ---------------------------------------------------------------- 档位阶梯

def tier_ladder(requested):
    """按"4K 优先"展开要试的档位序列。

    ⚠️ 起点永远是**请求档**（默认 4K），其余档位只在请求档确认拿不到画面后才轮到，
       且一旦落在低于请求档的位置，调用方必须把 `degraded` 上报给界面。
       —— 这就是用户口径"坚持 4K 优先不降级"的实现：可以试低档，但**不许静默**。
    """
    w, h = requested
    rest = [t for t in CAM_TIERS if t[1] < h] + [(640, 480)]
    out, seen = [], set()
    for t in [(w, h)] + rest:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------- 开流主逻辑

class OpenResult:
    """一次成功开流的完整说明（界面与日志都要看它）。

    ⚠️ `degraded` / `warnings` 不是装饰 —— 用户口径「坚持 4K 优先、不降级」的
       落地方式就是：**允许**在确认拿不到 4K 时退回低档，但**必须**让界面显示警告，
       绝不允许静默换成 480p。所以这两个字段是界面必读的。
    """

    __slots__ = ("cap", "camera", "backend", "width", "height", "fourcc",
                 "elapsed", "attempts", "degraded", "requested", "warnings")

    def __init__(self, cap, camera, backend, width, height, fourcc,
                 elapsed, attempts, degraded, requested, warnings):
        self.cap = cap
        self.camera = camera
        self.backend = backend
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self.elapsed = float(elapsed)
        self.attempts = list(attempts or [])
        self.degraded = bool(degraded)
        self.requested = tuple(requested)
        self.warnings = list(warnings or [])

    @property
    def mjpg(self):
        return _mjpg_ok(self.fourcc)

    def summary(self):
        note = f"（请求 {self.requested[0]}x{self.requested[1]}，已降级）" if self.degraded else ""
        fc = self.fourcc or "?"
        return (f"后端={self.backend} 索引={self.camera} 分辨率={self.width}x{self.height}"
                f" 格式={fc}{note} 用时={self.elapsed:.1f}s 尝试={len(self.attempts)} 次")


def open_camera_result(idx=None, backend=None, width=None, height=None,
                       budget=OPEN_BUDGET_SEC, on_event=None, profile=None,
                       on_progress=None):
    """打开摄像头，返回 `OpenResult`；失败抛 `CameraError`（不 sys.exit）。

    参数含义与重构前的 `open_camera` 一致（索引 / 后端 / 分辨率都不可靠，逐个试）。

    与旧实现的关键差别（这三条就是"加载很久"的解药）：
      ① **profile 先验后试**：上次成功的组合先试，命中即返回（常见路径 3s 级）。
      ② **同后端快速重试**：坏开流换**新句柄**重开同一组合，
         不再在重试之间夹子进程探测（旧实现实测白烧 16s）。
      ③ **全局时间预算** `budget`：到点就放弃并明确报错，不会出现 31s/48.9s。
         预算是**准入闸**（见 `MIN_ATTEMPT_SEC`）：剩余时间不够跑完一轮就不再开始，
         所以最坏情况只比预算晚一次读帧超时（≤2.5s），不会晚 7 秒。
      ④ 非首选后端**只探一次**且结果**落盘**（含 MJPG，见 `_probe_code`）。
      ⑤ `on_progress(backend, camera_index, (w, h), attempt)`：每次**起一轮开流之前**
         回调一次。为什么需要它：阶梯是采集线程内部的事，界面拿不到就只能干写
         "已经等了 N 秒"。有了它，家长能看到"正在试 DSHOW ／ 1280x720 第 2 轮"，
         "加载很久"到底是卡在探测、卡在坏开流、还是卡在等有效帧，一眼可辨。
         回调异常一律吞掉 —— 进度显示不该把开流搞失败。

    ⚠️ 本函数**必须在最终使用 cap 的那个线程里调用**（DSHOW/COM 绑定创建线程）。
       图形界面里这件事由 `CameraStream` 保证。
    """
    if cv2 is None:
        raise CameraError("缺少依赖：opencv", "请先执行： pip install mediapipe opencv-python")

    t_start = time.perf_counter()
    deadline = t_start + float(budget)
    prof = profile if profile is not None else CameraProfile.load()
    emit = on_event or (lambda msg: None)

    table = _backend_table()
    if backend and backend in table:
        order = [backend]
    else:
        # 首选后端 = 上次成功的那个（没有则按 DSHOW→MSMF→ANY 猜）
        order = list(BACKEND_NAMES)
        g = prof.last_good
        if g and g.get("backend") in table:
            order = [g["backend"]] + [b for b in order if b != g["backend"]]

    indices = [idx] if idx is not None else [0, 1, 2, 3]
    w_req = int(width or settings.DEFAULT_CAM_W)
    h_req = int(height or settings.DEFAULT_CAM_H)
    requested = (w_req, h_req)

    attempts = []
    skipped = set()        # 子进程判定不可用 / 会阻塞的后端
    absent = set()         # (后端, 索引) 确认设备不在，不必反复开

    def note(msg):
        try:
            emit(msg)
        except Exception:
            pass

    def progress(name, i, res, attempt):
        """把"正在试哪个组合"递出去（见参数 ⑤）；界面用它把等待时间说清楚。"""
        if on_progress is None:
            return
        try:
            on_progress(name, i, res, attempt)
        except Exception:
            pass

    # ---------- profile 的作用：决定**试的顺序**，不是"先试一遍再走阶梯"。
    #   ⚠️ 这里曾踩过一个反效果的设计：把 profile 做成独立的"阶段 0 预检"，
    #      于是同一个组合要先预检一次、再进阶梯一次。实测每次预检都要为坏开流
    #      白付 4~7s（本机 4K 坏开流：4 张全零帧 × ~1s，外加 set 的 2.6~4.5s），
    #      中位数被拖到 10.6s。改成"只调顺序"后：
    #        · `order[0]` = 上次成功的后端 → 首选后端**永不做子进程探测**
    #          （省掉的正是旧实现那 16s 的 MSMF/ANY 探测浪费）；
    #        · 档位阶梯本来就从这里的第一后端开始；
    #        · 上一次的成功记录只用来排序，不用来跳过验证 —— 换摄像头/被占用
    #          时仍然要靠"能否读到有效帧"说话。

    # ---------- 阶段 1：按档位阶梯 × 后端 × 同后端快速重试
    #   组织方式与旧实现的两处不同（都是"加载很久"的直接来源）：
    #     · 每一轮 attempt 只重开**还活着的**后端，且**不在重试之间插子进程探测**
    #       —— 旧实现每轮都把 MSMF/ANY 再探一遍，实测白烧 16s。
    #     · 探测结果落盘 + 内存缓存，一个组合一辈子最多付一次探测代价。
    for res in tier_ladder(requested):
        for attempt in range(MAX_OPEN_ATTEMPTS):
            remain = deadline - time.perf_counter()
            if remain < MIN_ATTEMPT_SEC:
                break
            alive = [n for n in order if n not in skipped]
            if not alive:
                break                      # 所有后端都排除了，阶梯再走也没意义
            any_candidate = False
            for name in alive:
                for i in indices:
                    if (name, i) in absent:
                        continue
                    remain = deadline - time.perf_counter()
                    if remain < MIN_ATTEMPT_SEC:
                        break              # 准入闸：来不及就别起了（见 MIN_ATTEMPT_SEC）
                    any_candidate = True
                    # ★ 先报名再动手：探测/开流/等有效帧都可能耗秒级，
                    #    界面必须知道"此刻在试哪个组合"（参数 ⑤）
                    progress(name, i, res, attempt)
                    # 非首选后端先做子进程可用性探测（防构造函数把主进程卡死）
                    if name != order[0]:
                        if not backend_viable_subprocess(
                                name, i, res[0], res[1],
                                timeout=min(OPEN_PROBE_TIMEOUT, max(1.0, remain)),
                                profile=prof):
                            attempts.append(
                                {"backend": name, "camera": i, "width": res[0],
                                 "height": res[1],
                                 "result": "子进程探测不可用/会阻塞（跳过该后端）",
                                 "sec": None})
                            skipped.add(name)
                            break
                    # ★ 在**调用线程**里创建（见文件头的 2x2 实测）
                    cap, info = open_once(i, name, res, deadline=deadline,
                                          timeout=min(2.5, max(0.5, remain)))
                    attempts.append(info)
                    if cap is not None:
                        w_act, h_act = info["actual_w"], info["actual_h"]
                        prof.set_tier(name, i, res[0], res[1], True, (w_act, h_act),
                                      info.get("fourcc"))
                        prof.set_last_good(name, i, w_act, h_act,
                                           info.get("fourcc"), info["sec"])
                        if info["sec"] > 6:
                            note(f"[摄像头] 这次开流偏慢（{info['sec']:.1f}s，"
                                 f"第 {attempt + 1} 轮）")
                        elif attempt:
                            note(f"[摄像头] 第 {attempt + 1} 轮重开才拿到画面")
                        return _result(cap, info, requested, t_start, attempts)

                    r = info.get("result", "")
                    if "打不开" in r:
                        absent.add((name, i))
                        continue
                    if "阻塞" in r:
                        skipped.add(name)  # 会卡死的后端：整个后端放弃，不再重试
                        break
                    # 全零/画面异常：同组合**换新句柄**再试一轮（下一轮 attempt）
                if name in skipped:
                    continue
            if not any_candidate:
                break                      # 该后端下所有索引都确认设备不在
        # 这一档试完了还没有效帧 —— 记下结论，往下退一档（会作为 degraded 上报）
        for name in order:
            if name in skipped:
                continue
            for i in indices:
                if (name, i) in absent:
                    continue
                prof.set_tier(name, i, res[0], res[1], False, None, "无有效画面")
        if deadline - time.perf_counter() < MIN_ATTEMPT_SEC:
            break

    # ---------- 失败：把"打不开"与"打开了但没画面"分清楚（两种病，处置不同）
    blank = [a for a in attempts if a.get("result") and
             ("全零" in a["result"] or "异常" in a["result"])]
    detail = _fmt_attempts(attempts)
    extra = ""
    if blank:
        extra = ("\n\n⚠️ 以下组合**能打开设备但读不到有效画面**："
                 "\n  " + "；".join(sorted({f"{a['backend']}/{a['camera']}/"
                                            f"{a['width']}x{a['height']}" for a in blank}))
                 + "\n  「画面全零」= 已自动重开仍失败，说明该档位确实取不到画面，"
                   "换 `--list-cameras` 看哪些档位 ✅。")
    raise CameraError(
        f"打不开任何摄像头（{time.perf_counter() - t_start:.0f}s 内没有找到能读到**有效画面**的组合）",
        (detail or "（没有尝试记录，可能设备完全不在）") + extra + "\n\n" + CHECKLIST
    )


def _result(cap, info, requested, t_start, attempts):
    """把 open_once 的原始信息整理成 OpenResult，并生成**必须让界面显示**的警告。"""
    w_act, h_act = info["actual_w"], info["actual_h"]
    degraded = (w_act, h_act) != tuple(requested)
    warns = []
    if degraded:
        warns.append(
            f"只拿到 {w_act}x{h_act}，低于请求的 {requested[0]}x{requested[1]}。"
            "存证照片清晰度会下降（判定角度不受影响）。"
            f"\n  先看是不是被别的程序占用，或用 --list-cameras 确认该档能否出画面；"
            "别把这事当正常。")
    if w_act >= 1280 and not _mjpg_ok(info.get("fourcc")):
        warns.append(
            f"该档位没协商到 MJPG（实际格式 {info.get('fourcc')}），"
            "而是未压缩的 YUY2 —— USB 2.0 带宽下高分辨率会又慢又不稳，"
            "这是驱动不提供该档 MJPG 的表现。")
    return OpenResult(cap=cap, camera=info["camera"], backend=info["backend"],
                      width=w_act, height=h_act, fourcc=info.get("fourcc"),
                      elapsed=time.perf_counter() - t_start, attempts=list(attempts),
                      degraded=degraded, requested=requested, warnings=warns)


def open_camera(idx=None, backend=None, width=None, height=None,
                budget=OPEN_BUDGET_SEC, on_event=None):
    """兼容旧签名：返回 `(cap, 实际索引)`；失败抛 `CameraError`。

    ⚠️ 新代码请用 `open_camera_result` —— 它带降级标记、耗时与警告，
       这些是"画面为什么糊/为什么慢"的唯一线索。本函数只为旧调用点与测试保留。
    """
    r = open_camera_result(idx=idx, backend=backend, width=width, height=height,
                           budget=budget, on_event=on_event)
    print(f"[摄像头] {r.summary()}")
    for w in r.warnings:
        print(f"[摄像头][警告] {w}")
    return r.cap, r.camera


# ---------------------------------------------------------------- 采集线程

class CameraStream:
    """**设备归属线程**：在一个长命线程里创建并持有 cap，只向外递最新帧。

    为什么必须这样（重构前引擎直接在推理线程里开+读，于是）：
      · 开流要 5~20s，这期间界面只会干等（用户报的"加载很久"最难忍的就是没有反馈）；
      · `cap.read()` 阻塞时整条推理循环一起卡住，界面跟着僵；
      · 摄像头被抢/拔线之后**没有任何恢复路径** —— 画面停在最后一帧，
        程序看起来还在跑，实际早就不更新了。

    本类的契约：
      · **创建与读取同线程**（DSHOW/COM 的硬要求，见文件头 2x2 实测）；
      · 帧槽 **latest-wins**（`latest()` 返回 `(frame, seq)`，取走即清空），
        所以下游 5 FPS 判定永远拿到最新一帧，也不会因为慢而堆帧；
      · `status()` 给界面实时进度：opening / ready / reconnecting / failed / stopped；
      · 掉线（超过 `STALL_SEC` 没有新帧）后自动 `open_camera_result` 重开，
        重开在**本线程**内进行，所以恢复出来的句柄仍然合法。

    ⚠️ 重开若失败，句柄被丢弃，界面显示"摄像头已断开"并继续按 `REOPEN_DELAY_SEC` 重试。
       不做"在别的线程里救回来"这种事 —— 那个句柄本来就绑在这个线程上，救不活。
    """

    def __init__(self, idx=None, backend=None, width=None, height=None,
                 budget=OPEN_BUDGET_SEC, on_event=None, name="camera"):
        self.idx = idx
        self.backend = backend
        self.width = width
        self.height = height
        self.budget = budget
        self._on_event = on_event or (lambda msg: None)
        self.name = name

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._cap = None
        self._frame = None
        self._seq = 0
        self._seen = 0                # 消费方游标（poll 用它实现"只给新帧"）
        self._last_frame_at = 0.0

        self._result = None           # OpenResult（成功后对外可查档位）
        self._phase = "idle"          # idle|opening|ready|reconnecting|failed|stopped
        self._short = None
        self._detail = None
        self._warns = []
        self._reopen_count = 0
        self._trying = None           # 正在试的组合（开流/重连期间才有值）

    # ---------------- 生命周期

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-stream",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        """请求停止。**不**在这里 release 句柄。

        ⚠️ 原因：cap 属于采集线程（COM 绑定）。若线程还活着就从别的线程 release，
           等于让两个线程同时碰同一个设备句柄 —— 实测这类"顺手清理"会偶发崩溃。
           线程若卡在 read 上，就让它带着句柄当 daemon 退出（进程结束时系统回收）。
        """
        self._stop.set()
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=3.0)     # 采集线程可能正阻塞在 read 上，只能等它自己回

    # ---------------- 对外只读接口（UI/推理线程调用，全部无阻塞）

    def latest(self):
        """返回当前帧槽 `(frame|None, seq)`。**不清空、不消费**。

        推理用这个（每个判定周期都要拿到"此刻最新的一帧"，哪怕上一周期已经看过它）；
        界面刷新用 `poll()`（只看新帧，省掉重复的 QImage 转换）。
        """
        with self._lock:
            return self._frame, self._seq

    def poll(self):
        """返回 (frame, seq)；与上次相同则 frame 为 None（省一次转换）。"""
        with self._lock:
            seq = self._seq
            if seq == self._seen:
                return None, seq
            self._seen = seq
            return self._frame, seq

    def status(self):
        with self._lock:
            return {"phase": self._phase, "short": self._short, "detail": self._detail,
                    "warnings": list(self._warns), "seq": self._seq,
                    "reconnects": self._reopen_count,
                    "result": self._result.summary() if self._result else None,
                    "width": getattr(self._result, "width", None),
                    "height": getattr(self._result, "height", None),
                    "backend": getattr(self._result, "backend", None),
                    "camera": getattr(self._result, "camera", None),
                    "degraded": getattr(self._result, "degraded", False),
                    "trying": dict(self._trying) if self._trying else None,
                    "stale_sec": (time.perf_counter() - self._last_frame_at)
                                 if self._last_frame_at else None}

    @property
    def result(self):
        return self._result

    # ---------------- 线程体

    def _set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _emit(self, msg):
        try:
            self._on_event(msg)
        except Exception:
            pass

    def _note_trying(self, name, i, res, attempt):
        """记录"此刻正在试哪个组合"（`open_camera_result` 的 on_progress）。

        只在 opening / reconnecting 期间有值，成功或失败即清空。
        界面靠它把"已经等了 8 秒"说成"已经等了 8 秒，正在试 MSMF ／ 1280x720 第 2 轮"
        —— 同样是等，能不能看出卡在哪儿，差别很大。
        """
        self._set(_trying={"backend": name, "camera": i,
                           "width": int(res[0]), "height": int(res[1]),
                           "attempt": int(attempt) + 1})

    def _run(self):
        """线程体：开流 → 采集循环 → （停摆则）重连。句柄只在本线程碰。"""
        # ★ 关键：开流发生在本线程 -> cap 与后续 read 同线程（COM 要求）
        self._set(_phase="opening", _short=None, _detail=None, _last_frame_at=time.perf_counter())
        self._emit(f"[采集] 正在打开摄像头（预算 {self.budget:.0f}s）…")
        try:
            try:
                res = open_camera_result(idx=self.idx, backend=self.backend,
                                         width=self.width, height=self.height,
                                         budget=self.budget, on_event=self._emit,
                                         on_progress=self._note_trying)
            except CameraError as e:
                self._set(_phase="failed", _short=e.short, _detail=e.detail, _trying=None)
                self._emit(f"[采集] 开流失败：{e.short}")
                return
            self._set(_cap=res.cap, _result=res, _phase="ready", _short=None,
                      _warns=list(res.warnings), _trying=None,
                      _last_frame_at=time.perf_counter())
            self._emit(f"[采集] {res.summary()}")
            for w in res.warnings:
                self._emit(f"[采集][警告] {w}")

            while not self._stop.is_set():
                ok, frame = read_frame(self._cap)      # 稳态：**不带超时**（见文档）
                if ok:
                    with self._lock:
                        self._frame = frame
                        self._seq += 1
                        self._last_frame_at = time.perf_counter()
                    if self._phase != "ready":
                        self._set(_phase="ready", _short=None)
                    continue

                # 读不到帧：可能设备被抢/掉线。看心跳决定要不要重开。
                stale = time.perf_counter() - self._last_frame_at
                if stale >= STALL_SEC:
                    self._emit(f"[采集] 已 {stale:.0f}s 没有新帧，尝试重连…")
                    if not self._reopen():
                        return
                else:
                    # ⚠️ 必须睡一下：否则"读不到但还没到停摆阈值"会变成空转打满 CPU
                    self._stop.wait(0.05)
            self._set(_phase="stopped")
        finally:
            # 线程退出时由**它自己**释放（别的线程不许碰这个句柄）
            self._release()

    def _reopen(self):
        """在同一线程里重开设备。成功返回 True；失败置 failed 并返回 False。"""
        self._release()
        self._set(_phase="reconnecting")
        while not self._stop.is_set():
            try:
                res = open_camera_result(idx=self.idx, backend=self.backend,
                                         width=self.width, height=self.height,
                                         budget=self.budget, on_event=self._emit,
                                         on_progress=self._note_trying)
            except CameraError as e:
                self._emit(f"[采集] 重连失败：{e.short}（{REOPEN_DELAY_SEC:.0f}s 后再试）")
                self._set(_short=e.short, _detail=e.detail, _trying=None)
                if self._stop.wait(REOPEN_DELAY_SEC):
                    self._set(_phase="stopped")
                    return False
                continue
            with self._lock:
                self._cap = res.cap
                self._result = res
                self._phase = "ready"
                self._short = None
                self._detail = None
                self._warns = list(res.warnings)
                self._reopen_count += 1
                self._trying = None
                self._last_frame_at = time.perf_counter()
            self._emit(f"[采集] 已重连：{res.summary()}")
            return True
        self._set(_phase="stopped")
        return False

    def _release(self):
        with self._lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


# ---------------------------------------------------------------- 视频源（离线回放）

class VideoSource:
    """把视频文件伪装成摄像头对象（接口与 cv2.VideoCapture 一致）。

    ⚠️ 它只换掉「帧从哪来」。**判定规则仍是同一份实现**
       （`metrics.compute_side_metrics` / `judge_side` / `ReminderGate` / 阈值），
       所以回放出来的时间线与实跑一致 —— 这正是当初把规则从界面里抽出来的原因。
    """

    def __init__(self, path):
        if cv2 is None:
            raise CameraError("缺少 opencv（cv2）", "无法回放视频。请先 pip install opencv-python")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise CameraError(f"打不开视频文件：{path}", "检查路径与文件是否完整。")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0     # 有些容器报 0/NaN
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.i = 0

    # ---- 与 VideoCapture 兼容的最小接口（read_frame / grab 都能直接用）
    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.i += 1
        return ok, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def set(self, *_a):
        return False                    # 文件没有"设分辨率"这回事

    def get(self, prop):
        return self.cap.get(prop)

    @property
    def vtime(self):
        """已读帧对应的**视频内时间**（秒）。回放的时间轴用它，不用墙上时钟。"""
        return self.i / self.fps

    @property
    def duration(self):
        return (self.frames / self.fps) if self.frames else 0.0


# ---------------------------------------------------------------- 排查用枚举

def list_cameras(width=None, height=None, out=print):
    """列出所有能打开的组合，并**逐档位标注是否真有画面**。

    ⚠️ 两条不可退让的要求（都是被实测定死的）：
      1) 必须判"有效画面"。只判 `cap.read()` 的 ok 会把"能打开但只返回全零占位帧"
         的档位报成「✅ 实际 1920x1080」—— 实测这颗摄像头在 1080p 下正是如此，
         排查时被这个假 ✅ 带偏，误以为摄像头正常。
      2) 必须**边测边 flush**，并且非首选后端先做**子进程探测**。
         本机 MSMF 的构造函数会**无限阻塞**，直接开会让 --list-cameras 永久挂住 ——
         而它正是文档里推荐的第一步排查手段，挂住最要命。
    """
    if cv2 is None:
        raise CameraError("缺少依赖：opencv", "请先执行： pip install mediapipe opencv-python")

    say = lambda s="": (out(s), sys.stdout.flush())

    say("枚举摄像头：后端 × 索引 × 分辨率档位（每档都验证是否真有画面）")
    say("说明：⚠️全零 = 该档位固件不支持；超时 = 该后端不可用；打不开 = 设备不在")
    say()

    if width and height:
        tiers = [(int(width), int(height))]
    else:
        tiers = CAM_TIERS + [(1600, 1200), (1280, 960), (1024, 768), (800, 600)]
        tiers = [t for i, t in enumerate(tiers) if t not in tiers[:i]]
        tiers.sort(key=lambda t: -(t[0] * t[1]))

    prof = CameraProfile.load()
    table = _backend_table()
    order = list(BACKEND_NAMES)
    g = prof.last_good
    if g and g.get("backend") in table:
        order = [g["backend"]] + [b for b in order if b != g["backend"]]

    usable, dead = [], set()
    for name in order:
        be = table[name]
        if name in dead:
            say(f"  后端={name}  跳过（前面已测出该后端会阻塞）")
            say()
            continue
        if name != order[0]:
            if not backend_viable_subprocess(name, 0, tiers[0][0], tiers[0][1], profile=prof):
                say(f"  后端={name}  跳过（子进程探测不可用/会阻塞）")
                dead.add(name)
                say()
                continue
        printed_any = False
        for i in range(3):
            head = False
            for (w, h) in tiers:
                cap = cv2.VideoCapture(i, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                if not head:
                    say(f"  后端={name}  索引={i}")
                    head = True
                    printed_any = True
                # 先 MJPG 后分辨率（同 open_once；驱动不支持的档会静默忽略）
                cap.set(cv2.CAP_PROP_FOURCC, MJPG_FOURCC)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                ok, frame, stat = wait_usable_frame(cap, tries=3, timeout=2.0,
                                                    blank_abort=warmup_budget(w, h))
                if ok:
                    ah, aw = frame.shape[:2]
                    note = "" if (aw, ah) == (w, h) else f"（实得 {aw}x{ah}）"
                    fc = _fourcc_str(cap)
                    say(f"    {w:>4}x{h:<4} ✅ 有画面  最大像素={stat['max_pixel']:>3}"
                        f"  格式={fc}{note}")
                    usable.append((name, i, aw, ah, fc))
                    prof.set_tier(name, i, w, h, True, (aw, ah), fc)
                elif stat["timed_out"]:
                    say(f"    {w:>4}x{h:<4} ⚠️ 后端阻塞超时（该后端不可用，跳过）")
                    cap.release()
                    dead.add(name)
                    break
                elif stat["max_pixel"] == 0:
                    say(f"    {w:>4}x{h:<4} ⚠️ 全零（该档位固件不支持）")
                    prof.set_tier(name, i, w, h, False, None, "全零")
                elif stat["max_pixel"] > 0:
                    say(f"    {w:>4}x{h:<4} ⚠️ 画面异常（最大像素 {stat['max_pixel']}）")
                else:
                    say(f"    {w:>4}x{h:<4} ⚠️ 读不到帧（可能被占用）")
                cap.release()
            if head:
                say()
        if not printed_any:
            say(f"  后端={name}  未枚举到设备")
            say()

    if not usable:
        say("未发现任何能读到**有效画面**的组合。")
        say("按上面的失败类型排查：全零=换档位；超时=换后端；打不开=查占用与隐私设置。")
        return []

    best = max(usable, key=lambda x: x[2] * x[3])
    say(f"可用组合 {len(usable)} 个（同一摄像头被多后端枚举到属正常）。")
    say(f"最高可用档位：{best[2]}x{best[3]}（后端={best[0]} 索引={best[1]} "
        f"格式={best[4]}）")
    req = (width or 3840, height or 2160)
    if (best[2], best[3]) != req:
        say(f"⚠️ 拿不到请求档位 {req[0]}x{req[1]}。本项目要求 4K 优先，"
            "若这是临时状况（如被占用），排除后重开；")
        say("   若是设备本身不支持，界面上会出现「已降级」警告，别当正常。")
    if best[2] * best[3] < 1280 * 720:
        say("⚠️ 没能拿到 720p。低分辨率下 1~1.5m 机位的关键点会明显抖动，")
        say("   建议把机位拉近，或换一颗支持 720p 以上 MJPG 的摄像头。")
    say()
    say("在 --camera 参数里用索引即可；若同一索引不同后端行为不一致，用 --backend 指定。")
    say(f"本项目默认档位 {CAM_TIERS[0][0]}x{CAM_TIERS[0][1]}（4K）；")
    say("若默认档位不在上面的 ✅ 列表里，用 --width/--height 显式指定一个 ✅ 的档位。")
    return usable

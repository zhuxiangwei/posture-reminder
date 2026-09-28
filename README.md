# 坐姿小卫士（Posture Reminder）

给家里的小学生做的**桌面坐姿提醒**：一颗摄像头侧放在孩子旁边监测坐姿，
连续歪着超过几秒就用语音提醒一句，坐正了完全不打扰。

**v1.0.0**　·　Windows　·　全程本地推理，画面不上传

---

## 这个程序的样子

双击 `启动坐姿小卫士.bat` 就开始了 —— **孩子不需要按任何按钮**。

- **零操作启动**：打开就是监测状态，首次启动自动采集 10 秒基线
- **先看脸色再看字**：状态用大色块 + 手绘表情（不用 emoji，避免不同 Windows 上变方块）
- **刻意不用红色**：警示用暖橙。红 = 惩罚感，会让孩子想关掉程序
- **只在该提醒时出声**：坐姿正常时**完全静音**（正反馈放在界面上，不用声音）
- **提醒有节奏**：30 秒内绝不重复提醒（哪怕这次的原因和上次不一样），
  同一个问题 90 秒内不重复叫

### 按键

| 键 | 作用 |
|---|---|
| **空格** | **暂停 / 继续** |
| 鼠标 | 底部「暂停」「声音」「重新学习坐姿」三个按钮 |

---

## 快速开始（3 步）

### 1. 建环境

```bash
:: 解释器用 Python 3.13（本项目不需要 tkinter，界面是 PySide6）
:: 本机可用：C:\Users\zxw\.workbuddy\binaries\python\versions\3.13.12\python.exe
python -m venv envs\posture
envs\posture\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple opencv-python==5.0.0.93 numpy==2.5.3 PySide6-Essentials==6.11.2
envs\posture\Scripts\python.exe -m pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple mediapipe==1.0.1
envs\posture\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.org/simple absl-py==2.5.0 certifi==2026.7.22 sounddevice==0.5.6 flatbuffers==25.12.19 matplotlib==3.11.2
```

> ⚠️ **`mediapipe` 必须加 `--no-deps`**：它声明依赖 `opencv-contrib-python`，
> 与 `opencv-python` 同时提供 `cv2`，pip 替换时会删除几千个文件 ——
> 在受限环境下会撞上批量删除护栏而中断。不装 contrib 也**实测能跑通**。
>
> ⚠️ **界面装 `PySide6-Essentials`，不要装 `PySide6` 元包**：6.x 起 PySide6 拆成
> Essentials + Addons 两个 wheel，且两者都写进同一个 `site-packages\PySide6\` 目录；
> Addons 覆盖 Essentials 的既有文件时会让 pip 连续删除，满 50 次就撞上同一个批量删除护栏，
> **安装会被中断、环境停在半装状态**。本项目只用 QtWidgets / QtGui / QtCore，Essentials 够用。
>
> ⚠️ 国内直连 PyPI 会很慢甚至卡死，务必带 `-i` 走镜像；但镜像上个别包
> （实测 `opencv-python==5.0.0.93`）会报 `from versions: none`，加 `--extra-index-url` 走官方兜底。
>
> ⚠️ **Windows 智能应用控制（SAC）可能拦 matplotlib 的 `_image.pyd`**（2026-09-28 实装踩到）。
> 症状是启动后崩在摄像头就绪之后，报 `AttributeError: 'NoneType' object has no attribute
> 'PoseLandmarker'`。真病因是 `mediapipe.tasks.python.vision` 整包导入失败：
> `ImportError: DLL load failed while importing _image: 应用程序控制策略已阻止此文件。`
> 该 pyd 本身完好（不是病毒/损坏）。**改名、换目录都没用**（SAC 按文件内容/签名判）。
>
> ✅ **但这个拦截很可能是临时的**：实测同一个字节完全相同的 `_image.pyd`
> （md5 `e43f68f3905a4006081f766dbc6f68ea` 前后一致）**重启一次电脑后就能正常导入**，
> 期间策略注册表值没变。CodeIntegrity 事件日志给出解释：SAC 的
> `VerifiedAndReputable Desktop` 策略在开机时被**刷新并激活**，带来更新的云信誉数据。
> → **遇到这个报错，先重启一次试试**，别急着关 SAC 或改代码。
>
> 项目另内置了一手兜底 `repos/p0-prototype/compat_matplotlib.py`：在 mediapipe 之前
> 用桩顶掉 matplotlib（mediapipe 只在 3D 调试可视化 `plot_landmarks` 里用它，推理管线不需要）。
> 真实 matplotlib 可用时它**完全不介入**，所以留着无害。
> 详见 `repos/p0-prototype/requirements.txt` 的「约束 4」。

> ⚠️ **换摄像头必须重新实测档位与后端**（2026-09-28 换摄像头时踩到）。
> 旧摄像头（Nebula 02）只有 MSMF 能用、DSHOW 全废、固定 1080p；
> 新摄像头（USB 2.0 Camera）**结论完全反转**：DSHOW 能用，而 MSMF
> **连构造函数都会无限阻塞**。照搬旧结论会让程序卡死或画面全黑。
> 换设备后先跑 `--list-cameras`（它会逐档标注是否真有画面）。详见下面的「实测数据」。
>
> 完整说明（含每一条的理由）见 `repos/p0-prototype/requirements.txt`。

### 2. 下模型

```bash
envs\posture\Scripts\python.exe tools\fetch_models.py
```

模型约 44MB，**没有放进仓库**（GitHub 单文件上限 100MB、超 50MB 会警告，
而且会永久拖大仓库体积），所以用脚本按需下载。

### 3. 跑

双击 **`启动坐姿小卫士.bat`**（无黑框）。

| 文件 | 用途 |
|---|---|
| `启动坐姿小卫士.bat` | 正常使用，`pythonw` 启动不弹控制台 |
| `演示模式.bat` | 不接摄像头，循环展示三种状态，语音也会播 —— **建议先点这个** |
| `调试模式.bat` | 打不开时用，保留控制台看报错 |

---

## 摆好机位：这一步最容易做错

**摄像头摆到孩子侧方 80~90°**（不是 45°），垫高 10~20cm 到接近眼睛高度，
距离 1~1.5m，台灯别变成侧逆光。

### ⚠️ 机位摆错的失败方式是「静默失效」

如果把摄像头正对孩子摆就跑，程序**不会报错、不会崩溃，而是永远不提醒**：
正面视角下同侧的耳和肩几乎落在一条竖直线上，`dx ≈ 0`，脖子前倾角恒接近 0。

所以摆完机位**必须验证**，别靠"看着像在跑"。打开
**「家长设置 → 机位检查」**，点「开始观察」，让孩子往前伸一下头再坐回去：

| 摆幅 | 判定 |
|---|---|
| ≥ 10° | ✅ 机位没问题 |
| 5~10° | ⚠️ 勉强可用，再往侧面转一点 |
| < 5° | ❌ 太正对孩子了，转到 80~90° 正侧 |

（判据用的是**摆幅**而不是当前值，所以**不用等标定跑完就能验机位**。）

---

## 能测什么 / 不能测什么

| | 说明 |
|---|---|
| ✅ 能测 | 脖子前倾、躯干前倾（塌腰）、低头（肩耳距离收缩） |
| ❌ 不能测 | 歪头、歪肩、离屏幕太近（这些需要正面视角，单侧面视角做不到） |

### 必须知道的限制

- **阈值还是未标定的初始值**。用几天后到「家长设置 → 审核判定」逐条审核程序判过的现场，
  它会用你的标注**反推建议阈值**。**误报比漏报致命** —— 叫得太勤孩子三天就关了
- **髋部被桌面挡住时**，塌腰测不出来（日志会提示），属正常
- 全程本地推理，**画面不上传任何地方**

---

## 隐私

程序会在「提醒时」和「每 3 分钟抽样一张」存下现场图到 `review/`，
用于事后审核并标定阈值。**这些图含孩子的影像**：

- 只存本机，不上传
- 上限 300 条，超出自动删最旧的（且优先删未审核的，保证你的标注不丢）
- 「审核判定」页有**一键清空**
- **`review/` 已在 `.gitignore` 里**，不会被提交到仓库

> 📷 **提醒现场会另外存一张原图**（按采集档取，现为 4K，约 1~3MB）。
> 这是 2026-09-26 按"处处用最高清"改的（原来为省空间是关的）——
> 原图也计入上面那 300 条上限，所以可留存的**条数**会少一些。
> 想省空间就在「家长设置 → 审核判定」里关掉「存原图」。

### 运行日志

程序会把运行日志写到 `repos/p0-prototype/logs/run-YYYYMMDD.log`（按天一个，**只留最近 7 天**）。
它是**纯文本**，不含画面；出问题（或复测）时把当天的日志文件发出来就能看到完整时间线。
不想要它就加 `--no-log` 启动，或把 `logs/` 直接删掉 —— 程序会自己重建。

---

## 项目结构

```
posture-reminder/
├── 启动坐姿小卫士.bat / 演示模式.bat / 调试模式.bat
├── docs/                          调研报告、部署任务书、方案
├── tools/
│   ├── fetch_models.py            下载模型权重
│   └── make_voice.py              生成语音片段（edge-tts → wav）
├── bench/models/                  模型权重（不入库）
└── repos/p0-prototype/
    ├── posture_monitor.py         判定逻辑 + 命令行
    ├── posture_engine.py          后台线程引擎（采集+推理+判定+迟滞+语音）
    ├── posture_app.py             PySide6 图形界面
    ├── tests/test_offline.py      156 个离线用例（不需要摄像头）
    └── voice/                     预生成语音（7 条 wav）
```

**三层分离**，关键理由是**迟滞规则只能有一份实现**：
CLI 和 GUI 各写一遍的话，「连续 4 秒才提醒」这类最容易毁产品的参数迟早会漂移。
所以抽成 `ReminderGate` 共用；推理也跑在独立线程里，否则界面会在最关键时刻卡死。

---

## 实测数据（本机 Windows）

| 项目 | 结果 |
|---|---|
| 摄像头 | USB **2.0 Camera**（VID_0C49&PID_636F，**标称 4K**，UVC），可用后端 **DSHOW**。旧摄像头 Nebula 02 已弃用/丢弃 |
| **★ 采集档位：3840×2160 + MJPG** | 逐档独立开合实测：**3840×2160 MJPG = 10/10 = 100% 可靠、~14 FPS**；2560×1440 MJPG 同为 10/10；640×480 = 15/15、~20 FPS |
| **★ 关键：压缩格式（FOURCC）** | OpenCV/DSHOW 默认协商 **YUY2 未压缩**，而 USB 2.0 实际带宽只有 ~35MB/s：480p YUY2=0.6MB/帧→20FPS 且 100% 可靠；**720p YUY2=1.8MB/帧→开流成功率仅 60%**；**1080p YUY2=4MB/帧→3.3 FPS、62%**。切 **MJPG** 后 **4K 反而 100% 可靠**。→ 设置顺序**必须**是「先 `CAP_PROP_FOURCC=MJPG`、再设分辨率」。这也解释了**为什么系统相机应用一直正常**：Media Foundation 会自动协商压缩格式 |
| **为什么不用 1080p** | 这颗驱动**只在高档位（4K/1440p）提供 MJPG**，1080p 请求 MJPG 会被静默忽略、退回 YUY2（于是只有 3.3 FPS、开流 62%）→ 1080p 反而不如 4K 好用 |
| **4K 全管线耗时** | 读帧(含 MJPG 解码) 35.3 + cvtColor 3.0 + 推理 14.2 = **52.6ms/帧**，5 FPS 预算 200ms → 占 1/4。实测吞吐 19 FPS |
| **开流会失败（防御仍在）** | 表现为**返回 ok=True 但内容全零的占位帧**（每帧固定 1000ms），**重开设备即可恢复**。程序内建重开重试（最多 5 轮）+ 有效帧判定，对低质量档位与将来换设备都是防线 |
| **"好的开流"前几帧也是全零** | 档位越高预热越长（480p 0 帧 / 720p 1 帧 / 1080p 3 帧）。判「有效画面」时必须容许预热帧，否则会把好开流误杀（这个坑真实踩过） |
| **MSMF 后端不可用** | `cv2.VideoCapture(0, CAP_MSMF)` **构造函数本身无限阻塞**（全新进程单独构造也复现）。注意：Windows 相机应用（用户态 Media Foundation）正常，是 **OpenCV 该后端与设备的组合问题**，不是摄像头不能跑 MF |
| **⚠️ 已更正的两处错误** | ① 曾写"480p 会把肩耳距离压到 32px 卡在门槛上"—— 错，门槛 `0.03×min(w,h)` **随分辨率缩放**，各档余量均 2.1~2.5x（320×240 都过筛）；② 曾写"1080p 是上限、720p 是甜点"—— 那是 **YUY2 未压缩**造成的假象，切 MJPG 后 4K 更可靠 |
| 推理耗时 | 4K 输入下模型推理 **14.2ms**（MediaPipe 输入固定 256×256）⚠️ 旧的"含人帧 19.2ms"来自**换摄像头前**的机器，可作参考 |
| 采样率 | 5 FPS，每帧 200ms 预算 → 余量充足 |
| **存证照片体积** | 4K + `review_keep_fullres=True` → 每张约 1~3MB，默认上限 300 条 → 最多约 1GB。磁盘紧张就在界面关掉"存原图" |
| 显示环境 | 逻辑桌面 **1920×1080**、可用区域 **1920×1032**、缩放 **2.00x**（`--screen` 实测）→ 高度充足，窗口不会被任务栏压住（笔记本时代"可用高度仅 680"那套不适用于本机） |

---

## 开发

```bash
:: 判定逻辑回归（145 用例，不需要摄像头、不需要模型）
envs\posture\Scripts\python.exe repos\p0-prototype\tests\test_offline.py

:: 界面冒烟自检（17 项，离屏跑，不会闪窗）
envs\posture\Scripts\python.exe repos\p0-prototype\posture_app.py --selftest

:: 看真实屏幕参数（HDPI 下窗口装不下时用它排查）
envs\posture\Scripts\python.exe repos\p0-prototype\posture_app.py --screen

:: 离线复现判定：拿一段录像把整条判定链重跑一遍（不接摄像头、不占人）
envs\posture\Scripts\python.exe repos\p0-prototype\posture_monitor.py --video 录像.mp4 --calib 10
```

⚠️ **别用 `--selftest` 查屏幕参数** —— 它强制走离屏平台，报的是 800×800 假数据。

### 文档

| 文档 | 内容 |
|---|---|
| `docs/01-research-report.md` | 调研报告：阈值依据、儿童场景研究、隐私考量（§6/§11 已过时） |
| `docs/02-deployment-taskbook.md` | 🚫 **已过时**：写的是笔记本部署，运行机已换成本机；其中的后端建议会致卡死 |
| `docs/03-side-only-plan.md` | **纯侧面方案 + 图形界面 + 审核标定**（当前主方案） |

> ⚠️ 关于**采集档位与后端**的一切结论，以本 README 的「实测数据」与
> `repos/p0-prototype/README.md` 为准 —— `docs/` 里的相关实测是旧摄像头的，已作废。

---

## 已知问题 / 路线

- [ ] **阈值标定**：需要真实使用几天积累审核样本（这是从"能跑"到"能用"的分水岭）
- [x] ~~含人帧大规模性能压测~~ → 2026-09-28 已在本机实测 **4K + MJPG 全管线**：
  读帧 35.3 + cvtColor 3.0 + 推理 **14.2ms** = **52.6ms/帧**，5 FPS 预算 200ms 只用掉 1/4。
  （不再需要"重跑 `--bench`"这条待办；若要复测，`--bench` 仍在）
- [x] ~~长期挂机稳定性~~ → 已用「视频回放喂给引擎」的方式压测（见 `--video`）
- [ ] 用 `--video` 拿真实录像复核阈值（误报/漏报都看得见，不用等现场）

---

## 许可

**暂未附许可协议** —— 没有 LICENSE 时默认「保留所有权利」。
如果希望别人能自由使用/修改，需要补一个（比如 MIT）。这个决定由仓库所有者来做。

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

> 📷 **提醒现场会另外存一张原图**（最高清档，1080p，约 150~500KB）。
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
    ├── tests/test_offline.py      88 个离线用例（不需要摄像头）
    └── voice/                     预生成语音（7 条 wav）
```

**三层分离**，关键理由是**迟滞规则只能有一份实现**：
CLI 和 GUI 各写一遍的话，「连续 4 秒才提醒」这类最容易毁产品的参数迟早会漂移。
所以抽成 `ReminderGate` 共用；推理也跑在独立线程里，否则界面会在最关键时刻卡死。

---

## 实测数据（本机 Windows）

| 项目 | 结果 |
|---|---|
| 摄像头 | USB **Nebula 02**（VID_3AAE&PID_6373），唯一可用后端 **MSMF**（DSHOW 按索引打不开）、固定 **1920×1080 @30fps**，不理会请求的分辨率 → 程序默认就按**最高清档**取 |
| 推理耗时 | 含人帧 **19.2ms**（无人帧 16.1ms —— **无人帧会偏乐观 40%**）⚠️ 这组来自**换摄像头前**的机器，且"16.1ms"与它自己标注的 40% 对不上（19.2/16.1 只有 +19%）→ **待你在本机重跑一次 `--bench` 复测**（需真人入画，见路线图） |
| 采样率 | 5 FPS，每帧 200ms 预算 → 余量充足 |
| **用最高清档的代价** | 受控 A/B（同一帧，同分辨率对照组漂移 ≤0.9%，n=120~150/档）：程序真实模式（VIDEO）下 **1920×1080 = 11.4ms vs 1280×720 = 10.5ms，只慢 7.8%（0.9ms）**；IMAGE 模式复测同向（20.7 vs 19.4ms，慢 6.5%）。5 FPS 下 1080p 只占预算 5.7% → **最高清档可以放心用**。（旧结论"降分辨率无效、只差 2.6%"来自把 512px 样图上采样到 640/1280 的测法，与本机真实 1080p 采集不可比） |
| 显示环境 | 逻辑桌面 **1920×1080**、可用区域 **1920×1032**、缩放 **2.00x**（`--screen` 实测）→ 高度充足，窗口不会被任务栏压住（笔记本时代"可用高度仅 680"那套不适用于本机） |

---

## 开发

```bash
:: 判定逻辑回归（103 用例，不需要摄像头、不需要模型）
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
| `docs/01-research-report.md` | 调研报告：阈值依据、儿童场景研究、隐私考量 |
| `docs/02-deployment-taskbook.md` | 部署任务书 |
| `docs/03-side-only-plan.md` | **纯侧面方案 + 图形界面 + 审核标定**（当前主方案） |

---

## 已知问题 / 路线

- [ ] **阈值标定**：需要真实使用几天积累审核样本（这是从"能跑"到"能用"的分水岭）
- [ ] 含人帧大规模性能压测（目前只测了单帧；换摄像头后**急需在本机重跑 `--bench`**）
- [x] ~~长期挂机稳定性~~ → 已用「视频回放喂给引擎」的方式压测（见 `--video`）
- [ ] 用 `--video` 拿真实录像复核阈值（误报/漏报都看得见，不用等现场）

---

## 许可

**暂未附许可协议** —— 没有 LICENSE 时默认「保留所有权利」。
如果希望别人能自由使用/修改，需要补一个（比如 MIT）。这个决定由仓库所有者来做。

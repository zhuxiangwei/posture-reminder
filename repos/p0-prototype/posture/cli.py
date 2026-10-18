# -*- coding: utf-8 -*-
"""命令行入口：`--list-cameras` / `--bench` / `--video` 回放 / 实时窗口。

原来这些都在 `posture_monitor.py` 里，和摄像头、指标、迟滞混在一个 1700 行的文件中。
现在本模块**只做"跟人多说话"这件事**：解析参数、开流、打印、收尾。
所有算法在 `metrics` / `gate`，所有设备细节在 `camera`，所有推理在 `pose`。

四种模式（互斥，按 `run()` 的分派顺序）：
    --list-cameras   纯排查，不加载模型
    --bench          推理性能自测（需要真人入画才有意义）
    --video FILE     离线回放整条判定链（不接摄像头、不占人在场时间）
    默认             标定 → 监测，开 OpenCV 预览窗（可用 --no-window 关掉）

★ 与图形界面（`posture.app`）的关系：
    判定规则、迟滞、阈值**完全共用**同一份实现，本模块只是另一层壳。
    所以"CLI 里看着对、界面里不对"这种问题结构上不可能出现 ——
    真出现过，就是因为早期把规则抄进了界面里。
"""

import argparse
import os
import statistics
import sys
import time

from posture import camera, deps, gate, metrics as mx, pose, settings, texts, voice
from posture.paths import DEFAULT_MODEL, find_model


# ---------------------------------------------------------------- 开流的小抄

def _open_for_cli(args, idx=None, label=""):
    """按命令行参数开一路摄像头，把 `OpenResult` 的关键信息打出来；失败直接退出。

    ⚠️ 统一走 `camera.open_camera_result`（而不是旧的 `open_camera`）：
       只有它带 `degraded` / `warnings`，而"坚持 4K 优先、任何降档必须显式警告"
       这条口径就落在这两个字段上。命令行里也不能静默降档。
    """
    try:
        res = camera.open_camera_result(
            idx=idx if idx is not None else args.camera,
            backend=args.backend, width=args.width, height=args.height,
            on_event=lambda m: print(m))
    except camera.CameraError as e:
        sys.exit(f"{label}{e.short}\n{e.detail}")
    print(f"{label}摄像头 {res.summary()}")
    for w in res.warnings:
        print(f"{label}⚠️ {w}")
    return res


# ---------------------------------------------------------------- 开流计时基准

OPEN_MEDIAN_MAX = 8.0
OPEN_P95_MAX = 15.0
"""「加载很久」的验收线（中位 ≤8s、p95 ≤15s）。

为什么必须写成数字：重构前这一步**没有上限**，实测中位 10.6s、最差 48.9s，
而且界面全程没有任何反馈 —— 用户只能反复重启。现在把它变成一条可复查的命令：

    python posture_monitor.py --bench-open 10

低于请求档时这里会直接打 `⚠️ 降档`，所以"有没有静默降档"也顺带验了。
"""


def bench_open(args):
    """连开 N 次摄像头，只统计**开流**耗时（不加载模型、不推理）。

    ⚠️ 为什么和 `--bench` 分开：`--bench` 测的是推理性能，而用户报的"加载很久"
       发生在开流这一步。两件事的修法完全不同（一个是模型/帧率，一个是后端协商），
       混在一个数字里就谁也修不好。

    每轮都完整走一遍 `open_camera_result`（构造期参数协商 → 有效帧验证 → 必要重开），
    开完**立刻 release** —— 不释放的话下一轮是被自己占着设备，测出来的数就没意义了。
    """
    import math

    deps.need_cv()
    n = max(1, int(args.bench_open))
    rounds, fails = [], 0
    print(f"[开流基准] 连续开 {n} 次（请求 {args.width}x{args.height}）…")
    for i in range(n):
        t0 = time.perf_counter()
        try:
            res = camera.open_camera_result(idx=args.camera, backend=args.backend,
                                            width=args.width, height=args.height)
            sec = time.perf_counter() - t0
            rounds.append(sec)
            note = "   ⚠️ 降档" if res.degraded else ""
            print(f"  第 {i + 1:2d} 轮 {sec:5.2f}s  {res.summary()}{note}")
            for w in res.warnings:
                print(f"          ⚠️ {w}")
            try:
                res.cap.release()
            except Exception:
                pass
        except camera.CameraError as e:
            rounds.append(time.perf_counter() - t0)
            fails += 1
            print(f"  第 {i + 1:2d} 轮  失败：{e.short}")
        time.sleep(0.6)      # 让驱动把上一轮的资源放干净，否则等待会被算进下一轮

    if not rounds:
        print("[开流基准] 一轮都没跑成")
        return 1
    srt = sorted(rounds)
    med = statistics.median(srt)
    p95 = srt[min(len(srt) - 1, max(0, math.ceil(0.95 * len(srt)) - 1))]   # 最近秩
    print(f"\n[开流基准] {len(rounds)} 轮：中位 {med:.2f}s　p95 {p95:.2f}s　"
          f"最差 {srt[-1]:.2f}s　失败 {fails} 次")
    print(f"           验收线：中位 ≤{OPEN_MEDIAN_MAX:.0f}s、p95 ≤{OPEN_P95_MAX:.0f}s"
          f"（重构前实测：中位 10.6s、最差 48.9s、无上限）")
    if fails:
        print("           ✗ 有开流失败的轮次")
        return 1
    if med > OPEN_MEDIAN_MAX or p95 > OPEN_P95_MAX:
        print("           ✗ 未达标")
        return 1
    print("           ✅ 达标")
    return 0


# ---------------------------------------------------------------- 基准模式

def bench(args):
    """测本机推理性能 —— 这是「本机能否本地胜任」的决定性测试。"""
    deps.need_cv()
    deps.need_mp()
    path = find_model(args.model)
    res = _open_for_cli(args)
    cap = res.cap
    print("[基准] 预热中 ...")
    for _ in range(5):
        camera.read_frame(cap)

    base = deps.mp_python.BaseOptions(model_asset_path=path)
    opts = deps.vision.PoseLandmarkerOptions(
        base_options=base, running_mode=deps.vision.RunningMode.VIDEO, num_poses=1
    )
    times, hit_t, miss_t, hits = [], [], [], 0
    N = args.bench_frames
    with deps.vision.PoseLandmarker.create_from_options(opts) as lm:
        t_start = time.perf_counter()
        for i in range(N):
            frame = camera.read_frame(cap)
            if frame is None:
                continue
            rgb = deps.cv2.cvtColor(frame, deps.cv2.COLOR_BGR2RGB)
            mi = deps.mp.Image(image_format=deps.mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            res_v = lm.detect_for_video(mi, int((time.perf_counter() - t_start) * 1000))
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            if res_v.pose_landmarks:
                hit_t.append(dt)
                hits += 1
            else:
                miss_t.append(dt)
    cap.release()

    def stat(a):
        if not a:
            return "无样本"
        a = sorted(a)
        return (f"n={len(a):3d}  mean={statistics.mean(a):6.1f}ms  "
                f"p95={a[min(len(a) - 1, int(len(a) * 0.95))]:6.1f}ms")

    rate = hits / max(1, len(times))
    mean_all = statistics.mean(times) if times else 0.0
    mean_hit = statistics.mean(hit_t) if hit_t else None
    mean_miss = statistics.mean(miss_t) if miss_t else None

    print(f"\n===== 基准结果（{os.path.basename(path)}）=====")
    print(f"  采样帧数     : {len(times)}")
    print(f"  检出成功率   : {hits}/{len(times)}  ({rate*100:.0f}%)")
    print()
    print(f"  【含人帧】   {stat(hit_t)}   <- 这才是真实成本")
    print(f"  【无人的帧】 {stat(miss_t)}")

    # MediaPipe 流水线 = 检测器找人框 -> 找到才跑关键点模型，所以两段成本可以相减拆开。
    # ⚠️ 但前提是「含人帧」确实含人。检出率低时那些帧多半是误检，
    #    「有人」样本本身不可信，拆出的比例也是假的 —— 实测踩过：
    #    误检帧给出「关键点仅占 7%」，而真人在图里的干净测量是 29%。差 4 倍。
    if rate < 0.5:
        print("\n  [成本拆解] 跳过 —— 检出率过低，『含人帧』可能多为误检，拆解结果不可信。")
    elif mean_hit and mean_miss:
        kp_share = (mean_hit - mean_miss) / mean_hit * 100
        print(f"\n  [成本拆解] 检测器 ≈ {mean_miss:.1f}ms（{100-kp_share:.0f}%），"
              f"关键点模型 ≈ {mean_hit-mean_miss:.1f}ms（{kp_share:.0f}%）")
        print(f"             -> ROI 跟踪跳过全图检测器可省掉其中约 {100-kp_share:.0f}%")

    # 关键防坑：检出率偏低时绝不能拿总均值下结论
    if not hit_t:
        print("\n  ⚠️ 整段采样一帧都没检测到人。")
        print("     此时跑的只是人体检测器，关键点模型全程未运行 ——")
        print("     本结果**不能代表真实推理成本**（会偏乐观约 40%）。")
        print("     请让一个人完整坐在摄像头前重跑。")
        return

    ref = mean_all
    if rate < 0.5:
        print(f"\n  ⚠️ 检出率仅 {rate*100:.0f}%，以下『总均值』不可用于结论！")
        print(f"     总均值 {mean_all:.1f}ms 被 {len(miss_t)} 个『画面里没人』的帧拉低了 "
              f"{(1 - mean_all / mean_hit) * 100:.0f}%（真实含人成本 {mean_hit:.1f}ms）。")
        print("     请让一个人完整坐在摄像头前重跑；本次结论已自动改用【含人帧】计算。")
        ref = mean_hit

    print(f"\n  ---- 结论（按 {ref:.1f}ms / 帧）----")
    print(f"  等效帧率   : {1000/ref:.1f} FPS")
    print(f"  判定需要的帧率仅 {settings.FPS_SAMPLE} FPS")
    ratio = (1000 / ref) / settings.FPS_SAMPLE
    verdict = "✅ 完全胜任" if ratio > 3 else ("⚠️ 勉强够用" if ratio > 1.5 else "❌ 建议降级")
    print(f"  余量       : {ratio:.1f}x  ->  {verdict}")
    print("\n把以上输出发回，即可确认是否需要启用降级阶梯。")


# ---------------------------------------------------------------- 机位装配

def build_cameras(args):
    """按模式装配机位：返回 [(cam_idx, metrics_fn, judge_fn, tag), ...]"""
    if args.side_only:
        if args.camera2 is not None:
            sys.exit("--side-only 与 --camera2 互斥：纯侧面模式只需要一路摄像头。\n"
                     "纯侧面请用： --side-only --camera <侧面机位索引>")
        return [(args.camera, mx.compute_side_metrics, mx.judge_side, "侧面")]
    cams = [(args.camera, mx.compute_metrics, mx.judge, "正面")]
    if args.camera2 is not None:
        cams.append((args.camera2, mx.compute_side_metrics, mx.judge_side, "侧面"))
    else:
        print("      提示：驼背/头前倾要靠侧面机位。")
        print("            只有侧面一颗摄像头 -> 用 --side-only --camera <索引>")
        print("            有正+侧两颗        -> 用 --camera <正面> --camera2 <侧面>")
    return cams


# ---------------------------------------------------------------- 实时模式

def main(args):
    """标定 → 监测，命令行版。

    ⚠️ 这里是**简单路径**：CLI 自己拿着 cap 逐帧 `grab()`。
       图形界面走的是 `camera.CameraStream`（独立采集线程 + 心跳重连）。
       为什么 CLI 不必上线程：它没有"界面会僵"这个问题，
       而且一次只跑几分钟、坏了直接重跑 —— 别为了统一而把命令行复杂化。
    """
    deps.need_cv()
    deps.need_mp()
    settings.apply_config(settings.load_config())        # 阈值以 config.json 为准
    mx.SIDE_YAW_DEG = args.side_yaw

    path = find_model(args.model)
    print(f"[模型] {path}")
    if args.side_only:
        print(f"[模式] 纯侧面单路（偏航角 {mx.SIDE_YAW_DEG:.0f}°）"
              " —— 测头前倾 / 塌腰 / 低头")
        if mx.SIDE_YAW_DEG < 70:
            print(f"      ⚠️ 建议把机位摆到 80~90°：{mx.SIDE_YAW_DEG:.0f}° 下前后位移只能看到 "
                  f"{mx._yaw_gain()*100:.0f}%，容易漏报。")

    specs = build_cameras(args)
    caps, lms, tags, fns, jfs = [], [], [], [], []

    print("[摄像头]")
    caps, results = [], []
    for cam_idx, mfn, jfn, tag in specs:
        try:
            res = camera.open_camera_result(
                idx=cam_idx, backend=args.backend,
                width=args.width, height=args.height,
                on_event=lambda m: print(m))
        except camera.CameraError as e:
            for c in caps:
                c.release()
            sys.exit(f"{tag}机位开流失败：{e.short}\n{e.detail}")
        results.append(res)
        caps.append(res.cap)
        print(f"      -> {tag}机位 {res.summary()}")
        for w in res.warnings:
            print(f"      ⚠️ {w}")
        fns.append(mfn)
        jfs.append(jfn)
        tags.append(tag)

    dual = len(caps) > 1
    for _ in caps:
        lms.append(pose.make_landmarker(path))

    interval = 1.0 / max(1, int(settings.FPS_SAMPLE))
    t0 = time.perf_counter()

    try:
        # ------------------------------ 阶段一：标定
        print(f"\n[标定] 请端坐保持 {args.calib:.0f} 秒，正在采集基线 ...")
        samples = [[] for _ in caps]
        last = [None] * len(caps)
        next_run = t0
        last_log = t0
        timed_out = False

        while True:
            now = time.perf_counter()
            if now >= next_run:
                next_run = now + interval
                for i, cap in enumerate(caps):
                    fr, m, _ = pose.grab(cap, lms[i], now - t0, fns[i])
                    if fr is not None:
                        last[i] = fr
                    if m:
                        samples[i].append(m)
                if now - last_log > 1.0:
                    last_log = now
                    print("  已采集：" + "  ".join(
                        f"{tags[i]}={len(samples[i])}" for i in range(len(caps))))
                if not args.no_window:
                    for i, fr in enumerate(last):
                        if fr is not None:
                            deps.cv2.imshow(f"Posture-{tags[i]}",
                                            pose.draw(fr.copy(), None,
                                                      f"CALIB {tags[i]} {len(samples[i])}",
                                                      (0, 200, 255)))
                    if deps.cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if min(len(s) for s in samples) >= 15 and now - t0 > args.calib:
                break
            # 兜底：任一路始终看不到人时，不能无限等下去
            if now - t0 > max(args.calib * 4, 60):
                timed_out = True
                break

        if timed_out or min(len(s) for s in samples) < 10:
            print("\n[标定失败] 各路样本数：" +
                  "  ".join(f"{tags[i]}={len(samples[i])}" for i in range(len(caps))))
            sys.exit("请确认每一路机位都拍到了完整上半身、光线充足，然后重试。")

        bases = [mx.build_baseline(s) for s in samples]
        for i in range(len(caps)):
            print(f"[标定完成] {tags[i]} 基线 = " +
                  ", ".join(f"{k}={v:.3f}" for k, v in sorted(bases[i].items())))
        print_sanity(tags, bases)

        # ------------------------------ 阶段二：监测
        print(f"\n[监测] 连续超标 {gate.SUSTAIN_SEC:.0f} 秒才提醒；按 q 退出。")
        if dual:
            print("        融合规则：正面「低头」+ 侧面「头前倾」=> 判为「含胸驼背」")
        else:
            print("        单侧面模式：独立判定 " + " / ".join(sorted(texts.SIDE_ISSUES)))
        print()

        rgate = gate.ReminderGate()   # 局部变量不叫 gate —— 那是模块名
        frames = 0
        paused = False                 # 空格键切换（与 GUI 同一套交互）
        pause_t0 = 0.0
        t_fps = time.perf_counter()
        next_run = time.perf_counter()
        if not args.no_window:
            print("窗口焦点在视频窗上时：空格 = 暂停 / 继续，q = 退出\n")

        while True:
            now = time.perf_counter()
            if now < next_run:
                if args.no_window:
                    time.sleep(0.003)
                else:
                    k = deps.cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord(" "):
                        paused = not paused
                        if paused:
                            pause_t0 = time.perf_counter()
                            print("⏸  已暂停（再按空格继续）")
                        else:
                            # ⚠️ 暂停期间 gate 一直没被 update()，恢复时 `now - since`
                            #    会突然变大，可能立刻误报一次 —— 所以必须 reset()。
                            #    MediaPipe 的时间戳不用管：暂停期间不调用推理，
                            #    恢复后 `now - t0` 只是往前跳了一段，仍是单调递增。
                            dt = time.perf_counter() - pause_t0
                            t_fps += dt        # FPS 统计里扣掉暂停时间
                            rgate.reset()
                            print("▶  已继续")
                continue
            next_run = now + interval
            frames += 1

            if paused:
                # 暂停：不推理、不判定，但继续读帧刷新画面，否则窗口会「无响应」
                for cap, tag in zip(caps, tags):
                    fr = camera.read_frame(cap)
                    if fr is not None and not args.no_window:
                        deps.cv2.imshow(f"Posture-{tag}",
                                        pose.draw(fr.copy(), None,
                                                  "PAUSED  (SPACE to resume)"))
                continue

            front_issues, side_issues, views = [], [], []
            for i, cap in enumerate(caps):
                fr, m, kps = pose.grab(cap, lms[i], now - t0, fns[i])
                if fr is None:
                    views.append((None, None, "读帧失败"))
                    continue
                if m is None:
                    views.append((fr, kps, "未检测到人"))
                    continue
                st = jfs[i](m, bases[i])
                if tags[i] == "正面":
                    front_issues = st
                else:
                    side_issues = st
                views.append((fr, kps, "坐姿良好" if not st else "！" + "/".join(st)))

            all_issues = (mx.fuse(front_issues, side_issues) if dual
                          else (front_issues or side_issues))

            # 迟滞：连续超标才提醒，持续良好才解除（逻辑在 gate，与 GUI 共用）
            st = rgate.update(all_issues, now)
            if st.fire:
                voice.play_reminder(st.fire)

            if not args.no_window:
                fps_now = frames / max(1e-6, now - t_fps)
                for i, (fr, kps, stt) in enumerate(views):
                    if fr is None:
                        continue
                    if stt == "坐姿良好":
                        col = (0, 200, 0)
                    elif stt.startswith("！"):
                        col = (0, 0, 255)
                    else:
                        col = (0, 200, 255)
                    sub = f"{fps_now:.1f} FPS"
                    if dual and all_issues:
                        sub += "  | 融合: " + "/".join(all_issues)
                    deps.cv2.imshow(f"Posture-{tags[i]}",
                                    pose.draw(fr.copy(), kps,
                                              f"{tags[i]}: {stt}", col, sub))
                if deps.cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for c in caps:
            c.release()
        if deps.cv2 is not None:
            deps.cv2.destroyAllWindows()

    print("[退出] 正常结束。")


# ---------------------------------------------------------------- 标定自检打印

def print_sanity(tags, bases):
    print("\n[标定自检]")
    for label, v, verdict, hint in mx.sanity_rows(tags, bases):
        mark = {"ok": "✅", "warn": "⚠️ ", "info": "ℹ️ "}[verdict]
        val = "（本帧拿不到）" if v is None else f"{v:.3f}"
        print(f"  {mark} {label:34s} = {val}")
        if verdict in ("warn", "info"):
            print(f"      {hint}")
    print("  提示：数值明显超出区间 -> 孩子标定时就没坐正，或机位/光照有问题，重新标。")


# ---------------------------------------------------------------- 视频回放

def _mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _fmt_metrics(m, b):
    """把当前指标与基线并排打出来，方便一眼看出"差多少才越界"。"""
    parts = []
    for k in ("neck_angle", "trunk_angle"):
        v = m.get(k)
        if v is None:
            continue
        bv = b.get(k)
        parts.append(f"{k}={v:.1f}°" + (f"(基线 {bv:.1f})" if bv is not None else ""))
    v = m.get("sh_ear_len")
    if v is not None and b.get("sh_ear_len"):
        parts.append(f"肩耳距={v:.0f}(基线 {b['sh_ear_len']:.0f})")
    return "  ".join(parts)


def _load_baseline_json():
    """读 baseline.json 里的 base 字段（与界面共用同一份基线，不含"壳"）。

    ⚠️ 这里**故意**不走 `settings.load_baseline(cfg)` —— 那条路径会校验
       摄像头/偏航角/分辨率是否和标定时一致，不一致就作废（实时监测必须如此，
       否则「低头」会静默漏报）。回放是离线复现，用户指定了就用现成的这份。
    """
    return settings.read_baseline_raw()


def replay(args):
    """离线回放一段视频，跑完整判定链并输出逐条时间线。

    为什么要有它：判定准不准（误报/漏报）以前只能"当场盯着看"。
    现在录一段视频就能离线重跑整条链路 —— **不占人在场的时间，而且可复现**
    （同一视频 + 同一阈值 → 同一时间线），用来调阈值比看回放录像快得多。

    用法：
        python posture_monitor.py --video rec.mp4 --calib 10   # 用视频前 10 秒标定
        python posture_monitor.py --video rec.mp4 --calib 0    # 用现有 baseline.json

    时间基准是**视频内时间**，不 sleep、不开窗，所以跑得比真实播放快很多。
    """
    deps.need_cv()
    deps.need_mp()
    settings.apply_config(settings.load_config())

    if args.camera2 is not None:
        sys.exit("[回放] --video 目前只支持纯侧面一路（判定规则与 --side-only 相同）；"
                 "请去掉 --camera2。")

    mx.SIDE_YAW_DEG = args.side_yaw

    try:
        src = camera.VideoSource(args.video)
    except camera.CameraError as e:
        sys.exit(f"{e.short}\n{e.detail}")
    print(f"[回放] {args.video}")
    print(f"        {src.frames} 帧 · {src.fps:.1f} FPS · 时长 {src.duration:.1f}s")
    lm = pose.make_landmarker(find_model(args.model))
    step = max(1, int(round(src.fps / max(1, int(settings.FPS_SAMPLE)))))  # 每 N 帧判定一次

    try:
        # ---------------- 标定：用视频前 calib 秒，或直接用现有 baseline
        if args.calib and args.calib > 0:
            t_end = min(src.duration, float(args.calib))
            samples = []
            while src.i < src.frames and src.vtime < t_end:
                if src.i % step != 0:
                    if not src.read()[0]:
                        break
                    continue
                fr, m, _ = pose.grab(src, lm, src.vtime, mx.compute_side_metrics)
                if fr is None:
                    break
                if m:
                    samples.append(m)
            if len(samples) < 10:
                sys.exit(f"[回放] 标定失败：前 {t_end:.0f}s 只采到 {len(samples)} 个有效样本。\n"
                         f"       检查这段视频里是否有人、光线是否够、机位是否侧面；\n"
                         f"       或改用 --calib 0（走已有的 baseline.json）。")
            base = mx.build_baseline(samples)
            print(f"[回放] 前 {t_end:.0f}s 标定基线：" +
                  ", ".join(f"{k}={v:.2f}" for k, v in sorted(base.items())))
        else:
            base = _load_baseline_json()
            if not base:
                sys.exit("[回放] --calib 0 需要 baseline.json：先用界面标定一次，"
                         "或改回 --calib 10。")
            print("[回放] 使用现有 baseline.json：" +
                  ", ".join(f"{k}={v:.2f}" for k, v in sorted(base.items())))

        # ---------------- 回放判定（迟滞与提醒间隔仍然走共用 ReminderGate）
        rgate = gate.ReminderGate()
        fired, counts = [], {}
        n_judge = n_skip = 0
        prev = []
        print("\n---- 判定时间线 ----")
        while src.i < src.frames - 1:
            if src.i % step != 0:
                if not src.read()[0]:
                    break
                continue
            t = src.vtime
            fr, m, _ = pose.grab(src, lm, t, mx.compute_side_metrics)
            if fr is None:
                break
            if m is None:
                n_skip += 1
                continue
            n_judge += 1
            issues = mx.judge_side(m, base)
            for it in issues:
                counts[it] = counts.get(it, 0) + 1
            st = rgate.update(issues, t)
            if issues and issues != prev:
                print(f"[{_mmss(t)}] 判定 → {'/'.join(issues)}    {_fmt_metrics(m, base)}")
            elif not issues and prev:
                print(f"[{_mmss(t)}] 恢复 → 坐姿良好")
            prev = list(issues)
            if st.fire:
                print(f"[{_mmss(t)}] ▶ 提醒「{st.fire}」")
                fired.append((t, st.fire))

        # ---------------- 汇总
        dur = max(1e-6, src.duration)
        print("\n---- 汇总 ----")
        print(f"  时长        {_mmss(src.duration)}")
        extra = f"（另有 {n_skip} 帧未检测到人，已跳过）" if n_skip else ""
        print(f"  判定帧      {n_judge} 帧（每 {step} 帧取 1 帧）{extra}")
        if fired:
            print(f"  提醒        {len(fired)} 次 / {dur / 60:.1f} 分钟"
                  f" → 平均每 {dur / len(fired):.0f} 秒一次")
        else:
            print("  提醒        0 次")
        if counts:
            for it, c in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  问题「{it}」  命中 {c} 个判定帧")
        else:
            print("  问题        没有任何问题命中（是全片都坐得好，还是阈值太松？）")
        print("\n  提示：误报看上面「判定 →」的行与录像是否对得上；"
              "漏报要看有没有该提醒却没出现「▶ 提醒」的片段。")
        return 0
    finally:
        src.release()


# ---------------------------------------------------------------- 参数表

def build_parser():
    ap = argparse.ArgumentParser(
        prog="posture_monitor",
        description="坐姿实时提醒 P0 原型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "常用组合：\n"
            "  纯侧面（纸笔作业推荐）  --side-only --camera 0 --calib 10\n"
            "  双机位                  --camera 0 --camera2 1 --calib 10\n"
            "  性能自测（需真人入画）  --bench --bench-frames 100\n"
            "  开流计时（加载很久？）  --bench-open 10\n"
            "  排查摄像头              --list-cameras\n"
            "  离线复现判定            --video 录像.mp4 --calib 10\n"
        ),
    )
    ap.add_argument("--bench", action="store_true", help="只测推理性能")
    ap.add_argument("--bench-frames", type=int, default=60, help="基准采样帧数")
    ap.add_argument("--bench-open", type=int, default=0, metavar="N",
                    help="只测**开流**耗时，连开 N 次并统计中位/p95（验收「加载很久」）")
    ap.add_argument("--list-cameras", action="store_true", help="列出可用摄像头（排查用）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="模型文件名")
    ap.add_argument("--side-only", action="store_true",
                    help="纯侧面单路模式（纸笔作业场景）：只做头前倾/塌腰/低头")
    ap.add_argument("--camera", type=int, default=None,
                    help="主路摄像头索引；--side-only 时指侧面机位")
    ap.add_argument("--camera2", type=int, default=None,
                    help="第二路（侧面）摄像头索引；给了就启用双机位。与 --side-only 互斥")
    ap.add_argument("--side-yaw", type=float, default=90.0,
                    help="侧向机位偏航角（度）：0=正对，90=正侧。默认 90，建议 80~90")
    ap.add_argument("--backend", choices=list(camera.BACKEND_NAMES), default=None,
                    help="强制指定采集后端；默认按上次成功的后端优先")
    ap.add_argument("--width", type=int, default=settings.DEFAULT_CAM_W,
                    help=f"采集宽度，默认 {settings.DEFAULT_CAM_W}"
                         f"（4K 优先；拿不到会明确警告，不会静默降档）")
    ap.add_argument("--height", type=int, default=settings.DEFAULT_CAM_H,
                    help=f"采集高度，默认 {settings.DEFAULT_CAM_H}")
    ap.add_argument("--calib", type=float, default=10.0, help="标定时长（秒）")
    ap.add_argument("--no-window", action="store_true", help="不开预览窗")
    ap.add_argument("--video", default=None,
                    help="离线回放：给定视频文件，跑完整判定链并输出逐条时间线（不接摄像头）")
    return ap


def run(argv=None):
    """命令行总入口（`posture_monitor.py` 只是它的一层薄壳）。"""
    args = build_parser().parse_args(argv)
    try:
        if args.list_cameras:
            camera.list_cameras(args.width, args.height)
            return 0
        if args.bench_open:
            return bench_open(args)
        if args.bench:
            bench(args)
            return 0
        if args.video:
            return replay(args)
        return main(args) or 0
    except camera.CameraError as e:
        # 摄像头错误在 CLI 里一律"说人话再退出"，不给用户看栈。
        sys.exit(f"{e.short}\n{e.detail}")


if __name__ == "__main__":
    run()

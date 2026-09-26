#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 MediaPipe 模型权重
=======================

模型文件共约 44MB（heavy 档单个就 29MB），不适合塞进 git 仓库
（GitHub 单文件上限 100MB、超 50MB 会警告，而且会永久拖大仓库体积）。
所以仓库里不存模型，用这个脚本按需下载。

    python tools/fetch_models.py              # 只下运行必需的（lite）
    python tools/fetch_models.py --all        # 全下（含 full / heavy，约 44MB）
    python tools/fetch_models.py --list       # 看看有哪些、下没下

⚠️ 一个必须处理的坑
------------------
模型 URL 里**档位名出现两次**：

    .../pose_landmarker/{档位}/float16/1/pose_landmarker_{档位}.task

漏掉下划线或写错档位名，最典型的结果是 **404**（实测如此）。
但项目调研笔记里还记了另一种情况：Google 返回 **NoSuchKey 的 XML 错误页、
而 HTTP 状态码却是 200** —— 这种情况下只看状态码就会把几百字节的 XML
当成模型存下来，然后报一个完全看不懂的加载错误。

所以本脚本用**两条校验**，一条挡已知情形、一条防御未知情形：
1. 文件头不能是 XML / HTML（防 200 + 错误页）
2. 大小必须 ≥ 1MB（最小的模型也有 5.5MB）

（第 1 条是防御性的：本次测试用坏 URL 触发的是 404，没复现出 200+XML。
  但成本极低，留着。）
"""
import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
MODEL_DIR = os.path.join(ROOT, "bench", "models")

BASE = "https://storage.googleapis.com/mediapipe-models"

# 档位名在 URL 里出现两次，这里用模板拼，避免手写出错
POSE_URL = BASE + "/pose_landmarker/pose_landmarker_{k}/float16/1/pose_landmarker_{k}.task"

MODELS = {
    # 文件名                          URL                                                   必需  大致大小
    "pose_landmarker_lite.task": (POSE_URL.format(k="lite"), True, 5.5),
    "pose_landmarker_full.task": (POSE_URL.format(k="full"), False, 9.0),
    "pose_landmarker_heavy.task": (POSE_URL.format(k="heavy"), False, 29.2),
}

MIN_BYTES = 1 * 1024 * 1024      # 最小的模型也有 5MB，低于 1MB 一定是错误页


def human(n):
    return f"{n / 1024 / 1024:.1f} MB"


def download(name, url):
    dst = os.path.join(MODEL_DIR, name)
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = dst + ".part"
    print(f"  下载 {name} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
            head = r.read(64)
            f.write(head)
            # ⚠️ 校验 1：错误页是 XML/HTML，正常模型是二进制 protobuf
            low = head.lstrip().lower()
            if low.startswith(b"<?xml") or low.startswith(b"<html") or low.startswith(b"<error"):
                os.remove(tmp)
                raise RuntimeError(
                    f"拿到的是错误页而不是模型（URL 可能写错了）：\n     {head[:120]!r}")
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(str(e))
    # ⚠️ 校验 2：大小下限
    size = os.path.getsize(tmp)
    if size < MIN_BYTES:
        os.remove(tmp)
        raise RuntimeError(f"文件只有 {human(size)}，远小于预期，判定为无效")
    os.replace(tmp, dst)
    print(f"    ✅ {human(size)}")


def main():
    ap = argparse.ArgumentParser(description="下载 MediaPipe 模型权重")
    ap.add_argument("--all", action="store_true", help="全部下载（含 full / heavy）")
    ap.add_argument("--list", action="store_true", help="列出状态后退出")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    if args.list:
        print(f"模型目录：{MODEL_DIR}\n")
        for name, (_url, need, mb) in sorted(MODELS.items()):
            p = os.path.join(MODEL_DIR, name)
            have = os.path.getsize(p) if os.path.isfile(p) else 0
            mark = f"✅ {human(have)}" if have >= MIN_BYTES else "—　未下载"
            flag = "必需" if need else "可选"
            print(f"  {name:30s} {flag}  约{mb:>5.1f}MB   {mark}")
        return 0

    targets = [n for n, (_u, need, _m) in MODELS.items() if args.all or need]
    print(f"模型目录：{MODEL_DIR}")
    print(f"准备下载 {len(targets)} 个\n")

    ok = 0
    for name in targets:
        url, _need, _mb = MODELS[name]
        dst = os.path.join(MODEL_DIR, name)
        if os.path.isfile(dst) and os.path.getsize(dst) >= MIN_BYTES and not args.force:
            print(f"  跳过 {name}（已存在 {human(os.path.getsize(dst))}）")
            ok += 1
            continue
        try:
            download(name, url)
            ok += 1
        except Exception as e:
            print(f"    ❌ {name} 失败：{e}")

    print(f"\n完成 {ok}/{len(targets)}。")
    if ok == len(targets):
        print("现在可以跑了：双击项目根目录的「启动坐姿小卫士.bat」")
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())

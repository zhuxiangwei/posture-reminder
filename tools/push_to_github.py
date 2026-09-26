#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把本地仓库推到 GitHub
=====================

本地已经做完：git 仓库已初始化、v1.0.0 已提交并打标签、远端已配置。
只差最后一步「推」。这个脚本负责把这一步做对 —— 包括处理一个常见的坑：

⚠️ **如果远端已经有提交，直接 push 会被拒。**
用网页建仓库时如果勾了「Add a README」，GitHub 会先建一个初始提交，
此时本地和远端是两段**无关历史**，普通 push 会报
`Updates were rejected... fetch first`。脚本会识别这种情况并给出两个选项，
让你自己选（而不是替你 force push 覆盖掉远端的东西）。

用法（只用标准库，项目环境或系统 Python 都可以）：
    envs\posture\Scripts\python.exe tools\push_to_github.py            # 推
    envs\posture\Scripts\python.exe tools\push_to_github.py --check    # 只看状态，不动手
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BRANCH = "main"
TAG = "v1.0.0"


def git(*args, check=False, capture=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=capture,
                       text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：\n{r.stderr or r.stdout}")
    return r


def main():
    ap = argparse.ArgumentParser(description="把本地仓库推到 GitHub")
    ap.add_argument("--check", action="store_true", help="只看状态，不推送")
    args = ap.parse_args()

    print("=" * 60)
    print("推到 GitHub：zhuxiangwei/posture-reminder")
    print("=" * 60)

    # ---- 0. 前置检查 ----
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        sys.exit("❌ 这里不是 git 仓库。")
    r = git("rev-parse", "--abbrev-ref", "HEAD")
    br = r.stdout.strip()
    if br != BRANCH:
        print(f"⚠️ 当前分支是 {br}，期望 {BRANCH}。切换：git branch -M {BRANCH}")

    if not git("rev-parse", "--verify", "HEAD").stdout.strip():
        sys.exit("❌ 还没有任何提交。")

    has_tag = bool(git("tag", "-l", TAG).stdout.strip())
    n_files = len(git("ls-files").stdout.strip().splitlines())
    subj = git("log", "-1", "--pretty=%s").stdout.strip()
    print(f"\n本地状态：")
    print(f"  分支      {br}")
    print(f"  文件数    {n_files}")
    print(f"  最新提交  {subj}")
    print(f"  标签      {TAG if has_tag else '（缺失！）'}")

    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        print(f"\n⚠️ 有未提交的改动（{len(dirty.splitlines())} 个文件）：")
        for l in dirty.splitlines()[:8]:
            print("    " + l)
        print("  这些不会被推送。要先提交就另开一个终端跑 git add / git commit。")

    # ---- 1. 看远端状态 ----
    print(f"\n正在读取远端状态 …")
    ls = git("ls-remote", "--heads", "origin")
    remote_ok = ls.returncode == 0
    if not remote_ok:
        print("  ⚠️ 读不到远端（网络/代理/权限问题），仍可尝试直接推。")
        remote_heads = {}
    else:
        remote_heads = {}
        for line in ls.stdout.strip().splitlines():
            if not line.strip():
                continue
            sha, ref = line.split("\t")
            remote_heads[ref.replace("refs/heads/", "")] = sha
        if remote_heads:
            print(f"  远端已有分支：{', '.join(remote_heads)}")
        else:
            print("  远端是空的（干净的首次推送）")

    # 判断远端是否已有内容、且和本地不是同一条历史
    need_force_hint = False
    if BRANCH in remote_heads:
        local_head = git("rev-parse", "HEAD").stdout.strip()
        if remote_heads[BRANCH] == local_head:
            print("\n✅ 远端已经是最新的，不用推。")
            return 0
        # 远端这个提交在本地历史里吗？不在就是两段无关历史
        anc = git("merge-base", "--is-ancestor", remote_heads[BRANCH], "HEAD")
        if anc.returncode != 0:
            need_force_hint = True

    if args.check:
        print("\n（--check 模式，不做任何推送）")
        return 0

    # ---- 2. 推主分支 ----
    print(f"\n推送 {BRANCH} …")
    p = git("push", "-u", "origin", BRANCH, capture=False)
    pushed = p.returncode == 0

    if not pushed and need_force_hint:
        print("""
────────────────────────────────────────────────────────────
❌ 推送被拒：远端已有提交，而且和本地不是同一条历史。

   最常见的原因：用网页建仓库时勾了「Add a README」，
   GitHub 先建了一个初始提交，于是本地和远端成了两段无关历史。

   两个选择，你自己定（脚本不替你决定）：

   ① 保留远端那个 README，把两边合起来（推荐，不丢东西）：
        git pull --rebase origin main --allow-unrelated-histories
        git push -u origin main

   ② 远端那个提交你不要了，用本地覆盖（会丢掉远端的提交）：
        git push -u origin main --force

   做之前可以先看一眼远端有什么：
        git fetch origin
        git log origin/main --oneline
────────────────────────────────────────────────────────────""")
        return 1

    if not pushed:
        print("""
推送失败。常见原因：
  · 没有配置凭据 —— 首次推送会弹浏览器让你登录 GitHub，跟着走完即可
  · 网络到不了 github.com（公司网络/代理）
  · 代理只放行了 api.github.com，没放行 github.com

   可以先用这条判断是不是网络问题：
        git ls-remote https://github.com/zhuxiangwei/posture-reminder
""")
        return 1

    # ---- 3. 推标签 ----
    if has_tag:
        print(f"\n推送标签 {TAG} …")
        t = git("push", "origin", TAG, capture=False)
        if t.returncode != 0:
            print(f"⚠️ 标签推送失败，可以单独补：git push origin {TAG}")
    else:
        print(f"\n⚠️ 本地没有 {TAG} 标签，跳过。")

    # ---- 4. 收尾 ----
    print(f"""
────────────────────────────────────────────────────────────
✅ 推送完成

   仓库：https://github.com/zhuxiangwei/posture-reminder
   标签：https://github.com/zhuxiangwei/posture-reminder/releases/tag/{TAG}

   建议顺手做两件事：
   1. 到仓库页面确认一下根目录有 README.md（会自动显示成首页）
   2. 想发正式 Release 的话，在标签旁边点「Create release」，
      把 CHANGELOG.md 里 v1.0.0 那段贴进说明即可
      （命令行发 Release 需要 gh CLI，本机没装）
────────────────────────────────────────────────────────────""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

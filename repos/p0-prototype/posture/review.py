# -*- coding: utf-8 -*-
"""判定现场留存与家长审核（`ReviewStore`）。

从 `posture_engine` 搬出来的原因：它是**数据采集**设施，不是运行时链路的一部分；
界面（审核面板、家长设置）要独立于引擎读它、算误报率、给阈值建议。

⚠️ 依赖写法：本模块用 `from posture import metrics as mx` 给指标模块起别名，
   因为 `save()` 的形参就叫 `metrics`（"这一帧的指标字典"是这个领域的正式名词），
   若直接 `from posture import metrics`，形参会在函数体内遮蔽模块 ——
   那种 bug 表现为"在 save() 里调用 metrics.side_score() 却拿到 dict"，很难查。
"""

import json
import os
import time

from posture import deps
from posture import metrics as mx
from posture import paths


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
        self.root = root or paths.REVIEW_DIR
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
        if not self.enabled or frame is None or deps.cv2 is None:
            return None
        try:
            self._ensure()
            t = time.time()
            ident = time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t*1000) % 1000:03d}"
            jpg = os.path.join(self.pending_dir, ident + ".jpg")
            ok, buf = deps.cv2.imencode(".jpg", frame,
                                        [int(deps.cv2.IMWRITE_JPEG_QUALITY), 82])
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
                    okf, buff = deps.cv2.imencode(
                        ".jpg", full_frame, [int(deps.cv2.IMWRITE_JPEG_QUALITY), 92])
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
                if mx.ISSUE_METRIC.get(d.get("issue")) != metric:
                    continue
                sc = mx.side_score(metric, d.get("metrics") or {}, d.get("baseline") or {})
                if sc is None:
                    continue
                (should if d["should_trigger"] else shouldnt).append(sc)
            cur = (cfg.get("side_thresholds") or {}).get(metric)
            res = mx.suggest_threshold(should, shouldnt, current=cur)
            if res:
                res["metric"] = metric
                out[metric] = res
        return out

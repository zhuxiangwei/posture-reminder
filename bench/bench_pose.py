import time, os, statistics, json
import numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SRC = "C:/Users/zxw/.workbuddy/binaries/python/envs/posture/Lib/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg"
img = cv2.imread(SRC)
print("源图尺寸:", img.shape if img is not None else None)

def run(model_name, path, frames, iters):
    base = mp_python.BaseOptions(model_asset_path=path)
    opts = vision.PoseLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        # 预热
        for _ in range(3):
            lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)))
        times, detected = [], 0
        for i in range(iters):
            f = frames[i % len(frames)]
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            res = lm.detect(mi)
            times.append((time.perf_counter() - t0) * 1000)
            if res.pose_landmarks: detected += 1
        times.sort()
        return {
            "model": model_name,
            "size_mb": round(os.path.getsize(path)/1024/1024, 2),
            "mean_ms": round(statistics.mean(times), 1),
            "p50_ms": round(times[len(times)//2], 1),
            "p95_ms": round(times[int(len(times)*0.95)], 1),
            "fps": round(1000/statistics.mean(times), 1),
            "detect_rate": f"{detected}/{iters}",
            "landmarks": len(res.pose_landmarks[0]) if res.pose_landmarks else 0,
        }

frames = []
for w in (640, 1280):
    h = int(img.shape[0] * w / img.shape[1])
    frames.append(cv2.resize(img, (w, h)))

results = []
for name in ("lite", "full", "heavy"):
    p = f"models/pose_landmarker_{name}.task"
    try:
        r = run(name, p, frames, 25)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))
    except Exception as e:
        print(f"{name} 失败: {type(e).__name__}: {e}")

json.dump(results, open("bench_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n结果已存 bench_result.json")
